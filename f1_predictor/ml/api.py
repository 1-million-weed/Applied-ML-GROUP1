from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, ClassVar, Any
import logging
import textwrap
from .api_pipeline import APIpipeline

from experimental.data_aquisition.openf1_get import OpenF1DataFetcher
from .dataset_manager import DatasetManager

import pandas 


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawInputData(BaseModel):
    """Schema for raw input data from the users."""
    MEETING_VALUES_DICT: ClassVar[dict[int, str]] = {
        1: "Australian GP",
        2: "Chinese Grand Prix",
        3: "Japanese Grand Prix",
        4: "Bahrain Grand Prix",
        5: "Saudi Arabian Grand Prix",
        6: "Miami Grand Prix",
        7: "Emilia Romagna Grand Prix",
        8: "Monaco Grand Prix",
    }

    current_lap: int = Field(..., ge=1, description="Current lap number during the race")
    meeting: int = Field(..., description="Meeting identifier - see MEETING_VALUES_DICT for options")

    class Config:
        schema_extra = {
            "example": {
                "current_lap": 45,
                "meeting": 1  # Australian GP
            }
        }


class DriverPrediction(BaseModel):
    """Individual driver prediction result."""
    position: int = Field(..., ge=1, le=20, description="Predicted final position (1-20)")
    racer_name: str = Field(..., description="Driver name")


class PredictionResponse(BaseModel):
    """Response schema for race predictions."""
    predictions: List[DriverPrediction] = Field(..., description="List of driver position predictions")
    race_info: Dict[str, Any] = Field(..., description="Race context information")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"position": 1, "racer_name": "Max Verstappen"},
                    {"position": 2, "racer_name": "Lewis Hamilton"}
                ],
                "race_info": {
                    "meeting_name": "Australian GP",
                    "current_lap": 45,
                    "total_laps": 58
                },
                "metadata": {
                    "prediction_timestamp": "2025-05-29T10:30:00Z",
                    "model_version": "1.0.0"
                }
            }
        }
        

class API:
    """F1 Race Prediction API Interface.

    Provides endpoints for predicting final race positions based on current race state.
    Uses a trained ML model to analyze current lap data and predict final driver standings.
    """

    def __init__(self, model) -> None:
        """Initialize the API wrapper for the trained F1 prediction model.

        Args:
            model: Trained model object with a `.predict()` method that returns
                  (positions, driver_names) tuple or similar structure.

        Raises:
            ValueError: If model doesn't have required predict method.
        """
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")

        self.model = model
        self.dataset_manager = DatasetManager()
        self.openf1 = OpenF1DataFetcher()

        # Initialize FastAPI app
        self.app = FastAPI(
            title="🏎️ F1 Race Predictor API",
            description=textwrap.dedent("""
            ## F1 Position Prediction API

            Predict final race positions for Formula 1 drivers based on current race conditions.

            ### Features:
            - **Real-time Predictions**: Get position forecasts during live races
            - **Multiple Circuits**: Support for 8 major F1 circuits (races done this year)

            ### Supported Meetings:
            - 1: Australian GP,
            - 2: Chinese Grand Prix,
            - 3: Japanese Grand Prix,
            - 4: Bahrain Grand Prix,
            - 5: Saudi Arabian Grand Prix,
            - 6: Miami Grand Prix,
            - 7: Emilia Romagna Grand Prix,
            - 8: Monaco Grand Prix,

            ### Usage:
            1. Select a meeting from the supported options (see `/meetings` endpoint)
            2. Provide current lap number
            3. Receive predicted final positions for all drivers

            **Note**: Predictions are made for the final race of the weekend (not practice/qualifying).
            """),
        )
        self._setup_routes()

    def _setup_routes(self):
        """Configure API routes and handlers."""

        @self.app.get("/", tags=["Info"])
        async def root():
            """API health check and basic information."""
            return {
                "message": "🏎️ F1 Race Predictor API",
                "status": "active",
                "supported_meetings": RawInputData.MEETING_VALUES_DICT
            }

        @self.app.get("/meetings", tags=["Info"])
        async def get_meetings():
            """Get list of supported F1 meetings/circuits."""
            return {
                "meetings": RawInputData.MEETING_VALUES_DICT,
                "total_count": len(RawInputData.MEETING_VALUES_DICT)
                # TODO: add more meet information
            }

        @self.app.get("/meetings={meeting_id}", tags=["Info", "query"])
        async def get_meeting_info(meeting_id: int):
            """
            Get detailed information about a specific F1 meeting/circuit.

            **Path Parameters**:
            - `meeting_id` (int): Identifier for the meeting (1–8)

            **Returns**:
            - Meeting name
            - Total laps for the circuit
            - Additional context (weather, track conditions, etc.)

            **Errors**:
            - `400`: Invalid meeting ID
            - `404`: Meeting not found

            **Example**:

            ```json
            GET /meetings/1
            {
              "meeting_name": "Australian GP",
              "total_laps": 58,
              "weather_conditions": "dry",
              "track_temperature": 35.2
            """
            if meeting_id not in RawInputData.MEETING_VALUES_DICT:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid meeting ID",
                        "valid_options": list(RawInputData.MEETING_VALUES_DICT.keys()),
                        "meeting_names": RawInputData.MEETING_VALUES_DICT
                    }
                )

            # Now, i need to use the openF1_get class to get more information about the meeting
            meeting_name = RawInputData.MEETING_VALUES_DICT[meeting_id]
            meeting_data = self.openf1.get_meeting_data(meeting_id) 


        @self.app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
        async def predict(data: RawInputData):
            """
                🎯 **Predict final race positions based on current race state**

                This endpoint analyzes the current race conditions and provides predictions
                for where each driver will finish in the final race standings.

                **Request Body**:
                - `current_lap` (int, optional): Current lap number (must be ≥ 1) (default: 1)
                - `meeting` (int, optional): Circuit identifier (1–8, see `/meetings` endpoint) (default: 1)

                **Returns**:
                A `PredictionResponse` object containing:
                - Final predicted positions for each driver
                - Race context information (circuit, lap, totals)
                - Prediction metadata (timestamp, model version, etc.)

                **Errors**:
                - `400`: Invalid meeting ID or lap number
                - `422`: Input validation failed
                - `500`: Internal server or prediction error

                **Example**:

                ```json
                POST /predict
                {
                  "current_lap": 45,
                  "meeting": 1
                }
                ```
            """
            return await self._handle_prediction(data)

    async def _handle_prediction(self, input_data: RawInputData) -> PredictionResponse:
        """Core prediction logic handler.

        Args:
            input_data: Validated input data from user

        Returns:
            PredictionResponse: Complete prediction results

        Raises:
            HTTPException: For various error conditions
        """
        try:
            logger.info(f"Processing prediction request: Meeting {input_data.meeting}, Lap {input_data.current_lap}")

            # Validate meeting selection
            if input_data.meeting not in RawInputData.MEETING_VALUES_DICT:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid meeting ID",
                        "provided": input_data.meeting,
                        "valid_options": list(RawInputData.MEETING_VALUES_DICT.keys()),
                        "meeting_names": RawInputData.MEETING_VALUES_DICT
                    }
                )

            # Extract user selections
            selected_meeting = input_data.meeting
            selected_lap = input_data.current_lap
            meeting_name = RawInputData.MEETING_VALUES_DICT[selected_meeting]

            logger.info(
                f"User selections - Meeting: {meeting_name}, Lap: {selected_lap}")
            self.api_pipeline = APIpipeline(self.model, selected_meeting, selected_lap)
            predictions = self.api_pipeline.run()
            

            

            

            

            # Build comprehensive response
            response = PredictionResponse(
                predictions=structured_predictions,
                race_info={
                    "meeting_id": selected_meeting,
                    "meeting_name": meeting_name,
                    "current_lap": selected_lap,
                    **race_context  # Additional context from dataset
                },
                metadata=self._generate_metadata()
            )

            logger.info(f"Prediction completed successfully for {meeting_name}")
            return response

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal prediction error",
                    "message": str(e),
                    "meeting": input_data.meeting,
                    "lap": input_data.current_lap
                }
            )


    def _process_model_output(self, raw_predictions: any) -> List[DriverPrediction]:
        """Convert raw model output to structured prediction objects.

        Args:
            raw_predictions: Raw output from model.predict()
                           Expected format: (positions, driver_names) or similar

        Returns:
            List of DriverPrediction objects with position and name
        """
        # TODO: Implement based on your model's output format

        # Example implementation - replace with your actual model output processing:
        predictions = []

        # Assuming raw_predictions is a tuple of (positions, driver_names)
        # Adjust based on your actual model output structure
        try:
            positions, driver_names = raw_predictions  # TODO: Update based on actual output format

            for position, driver_name in zip(positions, driver_names):

                predictions.append(DriverPrediction(
                    position=int(position),
                    racer_name=str(driver_name)
                ))

        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            # TODO: Add fallback handling or re-raise with more context
            raise ValueError(f"Failed to process model predictions: {e}")

        # Sort by predicted position
        predictions.sort(key=lambda x: x.position)
        return predictions

    def _generate_metadata(self) -> Dict:
        """Generate prediction metadata.

        Returns:
            Dictionary with prediction metadata
        """
        from datetime import datetime

        return {
            "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": "1.0.0",  # TODO: Use actual model version
            "api_version": "1.0.0",
            # TODO: Add more metadata as needed:
            # - model training date
            # - feature importance
            # - data sources used
        }

