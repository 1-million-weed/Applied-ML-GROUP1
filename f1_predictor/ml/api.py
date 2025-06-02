from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, ClassVar, Any
import logging
import textwrap
from .api_pipeline import APIpipeline, CurrentYearInfo
from ..models.model import Model


# TODO: add year input for api.
# TODO: meeting cant be zero

class RawInputData(BaseModel):
    """Schema for raw input data from the users.

    - :cvar MEETING_VALUES_DICT: Mapping of meeting IDs to race names
    - :vartype MEETING_VALUES_DICT: dict[int, str]
    """
    @classmethod
    def get_full_meeting_names(cls) -> dict[str, int]:
        """Get full meeting names from CurrentYearInfo."""
        current_year_info = CurrentYearInfo()
        return current_year_info.list_rounds()
    
    @classmethod
    def get_meeting_id_to_name(cls) -> dict[int, str]:
        """Get mapping from meeting ID to full race name."""
        # Get full names and create inverted dictionary (ID → name)
        full_names = cls.get_full_meeting_names()
        return {meeting_id: race_name for race_name, meeting_id in full_names.items()}
    
    # Replace direct initialization with class properties
    @property
    def FULL_MEETING_NAMES_DICT(self) -> dict[str, int]:
        return self.get_full_meeting_names()
        
    @property
    def MEETING_VALUES_DICT(self) -> dict[int, str]:
        return self.get_meeting_id_to_name()
    
    # Add class-level properties that can be accessed without instance
    @classmethod
    @property
    def FULL_MEETING_NAMES_DICT_CLASS(cls) -> dict[str, int]:
        return cls.get_full_meeting_names()
        
    @classmethod
    @property
    def MEETING_VALUES_DICT_CLASS(cls) -> dict[int, str]:
        return cls.get_meeting_id_to_name()

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
    """Individual driver prediction result.

    - :param position: Predicted final position (1-20)
    - :type position: int
    - :param racer_id: Driver Id
    - :type racer_id: int
    """
    position: int = Field(..., ge=1, le=20, description="Predicted final position (1-20)")
    racer_id: int = Field(..., description="Driver Id")


class PredictionResponse(BaseModel):
    """Response schema for race predictions.

    - :param predictions: List of driver position predictions
    - :type predictions: List[DriverPrediction]
    - :param race_info: Race context information
    - :type race_info: Dict[str, Any]
    - :param metadata: Prediction metadata
    - :type metadata: Dict[str, Any]
    """
    predictions: List[DriverPrediction] = Field(..., description="List of driver position predictions")
    race_info: Dict[str, Any] = Field(..., description="Race context information")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"position": 1, "racer_id": "1"},
                    {"position": 2, "racer_id": "44"}
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

    def __init__(self, model: "Model") -> None:
        """Initialize the API wrapper for the trained F1 prediction model.

        :param model: Trained model object with a `.predict()` method that returns
                     (positions, driver_names) tuple or similar structure
        :type model: Model
        :raises ValueError: If model doesn't have required predict method
        """
        self.logger = logging.getLogger(__name__)
        if not hasattr(model, 'predict'):
            self.logger.error(f"Provided model ({model}) does not have a 'predict' method")
            raise ValueError("Model must have a 'predict' method")

        self.model = model
        meeting_names = RawInputData.get_full_meeting_names()
        self.logger.info("Initializing FastAPI with model: %s", model.__class__.__name__)
        self.app = FastAPI(
            title="🏎️ F1 Race Predictor API",
            description=textwrap.dedent(f"""
            ## F1 Position Prediction API

            Predict final race positions for Formula 1 drivers based on current race conditions.

            ### Features:
            - **Real-time Predictions**: Get position forecasts during live races
            - **Multiple Circuits**: Support for 8 major F1 circuits (races done this year)

            ### Supported Meetings:
            {meeting_names}

            ### Usage:
            1. Select a meeting from the supported options (see `/meetings` endpoint)
            2. Provide current lap number
            3. Receive predicted final positions for all drivers

            **Note**: Predictions are made for the final race of the weekend (not practice/qualifying).
            """),
        )
        self._setup_routes()

    def _setup_routes(self):
        """Configure API routes and handlers.

        Sets up FastAPI endpoints for health checks, meeting information, and predictions.
        """

        @self.app.get("/", tags=["Info"])
        async def root():
            """API health check and basic information.

            - :return: API status and supported meetings
            - :rtype: dict
            """
            self.logger.info("API root endpoint accessed")
            return {
                "message": "🏎️ F1 Race Predictor API",
                "status": "active",
                "supported_meetings": RawInputData.get_meeting_id_to_name()
            }

        @self.app.get("/meetings", tags=["Info"])
        async def get_meetings():
            """Get list of supported F1 meetings/circuits.

            - :return: Dictionary containing meeting mappings and count
            - :rtype: dict
            """
            self.logger.info("Fetching supported meetings")
            meeting_dict = RawInputData.get_meeting_id_to_name()
            return {
                "meetings": meeting_dict,
                "total_count": len(meeting_dict)
                # TODO: add more meet information
            }
            

        @self.app.get("/meetings/{meeting_id}/max-laps", tags=["Info"])
        async def get_max_laps(meeting_id: int):
            """Get the maximum number of laps for a specific meeting/round.
            
            - :param meeting_id: The ID of the meeting/round
            - :type meeting_id: int
            - :return: The maximum number of laps for the meeting
            - :rtype: dict
            - :raises HTTPException: 404 if the meeting ID is not found
            """
            self.logger.info(f"Fetching max laps for meeting ID: {meeting_id}")
            try:
                current_year_info = CurrentYearInfo()
                max_laps = current_year_info.get_max_laps_round(meeting_id)
                
                # Get the meeting name if available
                meeting_dict = RawInputData.get_meeting_id_to_name()
                meeting_name = meeting_dict.get(meeting_id, "Unknown Meeting")
                
                return {
                    "meeting_id": meeting_id,
                    "meeting_name": meeting_name,
                    "max_laps": max_laps
                }
            except ValueError as e:
                self.logger.error(f"Error fetching max laps: {str(e)}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Meeting not found",
                        "message": str(e)
                    }
                )

        @self.app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
        async def predict(data: RawInputData):
            """Predict final race positions based on current race state.

            This endpoint analyzes the current race conditions and provides predictions
            for where each driver will finish in the final race standings.

            - :param data: Input data containing current lap and meeting information
            - :type data: RawInputData
            - :return: Prediction response with driver positions and metadata
            - :rtype: PredictionResponse
            - :raises HTTPException: 400 for invalid input, 422 for validation errors, 500 for server errors

            **Request Body**:

            - `current_lap` (int): Current lap number (must be ≥ 1)
            - `meeting` (int): Circuit identifier (1–8, see `/meetings` endpoint)

            **Returns**:

            A `PredictionResponse` object containing:

            - Final predicted positions for each driver
            - Race context information (circuit, lap, totals)
            - Prediction metadata (timestamp, model version, etc.)

            **Example**::

                POST /predict
                {
                  "current_lap": 45,
                  "meeting": 1
                }
            """
            self.logger.info("Received prediction request", extra={"data": data})
            return await self._handle_prediction(data)

    async def _handle_prediction(self, input_data: RawInputData) -> PredictionResponse:
        """Handle the prediction logic for race position forecasting.

        :param input_data: Validated input data from the API request
        :type input_data: RawInputData
        :return: Complete prediction response with driver positions and metadata
        :rtype: PredictionResponse
        :raises HTTPException: For validation errors or internal prediction failures
        """
        try:
            print(input_data.meeting, input_data.current_lap)
            # Validate input data
            self.validate_input_data(input_data)

            # Extract user selections
            selected_meeting = input_data.meeting
            selected_lap = input_data.current_lap
            meeting_name = RawInputData.get_meeting_id_to_name()[selected_meeting]

            pipeline = APIpipeline(self.model, selected_meeting, selected_lap)

            # Raw predictions are in format {driver_id: position}
            raw_predictions_dict = pipeline.run()

            self.logger.info("Received raw predictions", extra=raw_predictions_dict)
            
            # Convert dictionary to list of DriverPrediction objects
            driver_predictions = []
            for driver_id, position in raw_predictions_dict.items():
                # Convert position to integer if it's a numpy type or other
                position_int = int(position)
                
                driver_predictions.append(DriverPrediction(
                    position=position_int,
                    racer_id=str(driver_id)
                ))
            
            # Sort predictions by position
            driver_predictions.sort(key=lambda x: x.position)

            # Build comprehensive response
            response = PredictionResponse(
                predictions=driver_predictions,
                race_info={
                    "meeting_id": selected_meeting,
                    "meeting_name": meeting_name,
                    "current_lap": selected_lap,
                },
                metadata=self._generate_metadata()
            )

            return response

        except HTTPException as http_exc:
            # Re-raise HTTP exceptions as-is
            self.logger.error("HTTPException occurred during prediction", str(http_exc.detail), extra={
                "meeting": input_data.meeting,
                "lap": input_data.current_lap
            })
            raise
        except Exception as e:
            self.logger.error("Unexpected error during prediction", str(e), extra={
                "meeting": input_data.meeting,
                "lap": input_data.current_lap
            })
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal prediction error",
                    "message": str(e),
                    "meeting": input_data.meeting,
                    "lap": input_data.current_lap
                }
            )

    def _generate_metadata(self) -> Dict:
        """Generate prediction metadata.

        :return: Dictionary with prediction metadata including timestamp and version info
        :rtype: Dict[str, Any]

        .. note::
           Currently includes prediction timestamp, model version, and API version.
           Future versions may include training date and feature importance.
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

    def validate_input_data(self, input_data):
        if input_data.current_lap < 1: # TODO:check upper limit
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid lap number",
                    "provided": input_data.current_lap,
                    "valid_range": "1 or greater"
                }
            )
        
        meeting_dict = RawInputData.get_meeting_id_to_name()
        if input_data.meeting not in meeting_dict:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid meeting ID",
                    "provided": input_data.meeting,
                    "valid_options": list(meeting_dict.keys()),
                    "meeting_names": meeting_dict
                }
            )

        # Check if the current lap exceeds max laps for the selected meeting
        try:
            current_year_info = CurrentYearInfo()
            max_laps = current_year_info.get_max_laps_round(input_data.meeting)
            
            if input_data.current_lap > max_laps:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid lap number",
                        "provided": input_data.current_lap,
                        "max_laps": max_laps,
                        "message": f"The selected meeting only has {max_laps} laps."
                    }
                )
        except ValueError:
            # If we can't fetch max laps, just skip this validation
            pass

