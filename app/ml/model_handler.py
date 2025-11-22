"""
Enhanced ML model handler for ECGWeb2 V2
"""
import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from flask import current_app


class ECGAnalyzer:
    """ECG Analysis handler"""

    BEAT_TYPES = {0: "Normal", 1: "Abnormal"}

    def __init__(self, model_path=None):
        """Initialize analyzer with model"""
        self.model_path = model_path or current_app.config['MODEL_PATH']
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the ML model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            self.model = joblib.load(self.model_path)
            current_app.logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            current_app.logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_ecg_graph(self, data_row, beat_index, output_path):
        """
        Generate ECG visualization for a single beat

        Args:
            data_row: Series containing ECG data points
            beat_index: Index of the beat
            output_path: Full path where to save the graph

        Returns:
            str: Path to the saved graph
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Set up the ECG grid
            ax.set_xlim(0, 190)
            ax.set_ylim(0, 1)

            # Major and minor ticks
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(1))
            ax.yaxis.set_minor_locator(AutoMinorLocator(0.02))

            # Grid styling
            ax.grid(which='major', color='#2a3439', linewidth=1.5)
            ax.grid(which='minor', color='#000000', linestyle=':', linewidth=0.5)

            # Remove x-axis labels for cleaner look
            ax.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False
            )

            # Plot ECG signal
            ax.plot(data_row[:186], color='#e74c3c', linewidth=2)

            # Add title
            ax.set_title(f'ECG Beat #{beat_index}', fontsize=14, fontweight='bold', pad=20)

            # Save the figure
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return output_path

        except Exception as e:
            current_app.logger.error(f"Error generating graph for beat {beat_index}: {str(e)}")
            plt.close('all')
            raise

    def predict(self, data):
        """
        Make predictions on ECG data

        Args:
            data: DataFrame containing ECG data

        Returns:
            tuple: (predictions, probabilities)
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Make predictions
            predictions = self.model.predict(data)
            probabilities = self.model.predict_proba(data)

            # Extract confidence scores
            confidence_scores = [round(max(prob), 4) for prob in probabilities]

            # Convert predictions to labels
            prediction_labels = [self.BEAT_TYPES.get(pred, "Unknown") for pred in predictions]

            return prediction_labels, confidence_scores

        except Exception as e:
            current_app.logger.error(f"Error making predictions: {str(e)}")
            raise

    def analyze_file(self, csv_path, analysis_id=None):
        """
        Analyze an entire CSV file

        Args:
            csv_path: Path to the CSV file
            analysis_id: Optional analysis ID for naming graphs

        Returns:
            dict: Analysis results containing predictions, graphs, and statistics
        """
        try:
            # Read data
            current_app.logger.info(f"Reading data from {csv_path}")
            data = pd.read_csv(csv_path)

            if len(data.columns) < 186:
                raise ValueError(f"Invalid CSV format. Expected at least 186 columns, got {len(data.columns)}")

            # Generate graphs
            graphs_folder = current_app.config['GRAPHS_FOLDER']
            graph_paths = []

            current_app.logger.info(f"Generating {len(data)} ECG graphs...")
            for i in range(len(data)):
                graph_filename = f"analysis_{analysis_id or 'temp'}_beat_{i}.png"
                graph_path = os.path.join(graphs_folder, graph_filename)
                self.generate_ecg_graph(data.iloc[i], i, graph_path)
                graph_paths.append(graph_filename)

            # Make predictions
            current_app.logger.info(f"Making predictions on {len(data)} beats...")
            predictions, confidences = self.predict(data)

            # Calculate statistics
            normal_count = predictions.count("Normal")
            abnormal_count = predictions.count("Abnormal")

            results = {
                'total_beats': len(data),
                'normal_count': normal_count,
                'abnormal_count': abnormal_count,
                'predictions': predictions,
                'confidences': confidences,
                'graph_paths': graph_paths,
                'data': data
            }

            current_app.logger.info(f"Analysis complete: {normal_count} normal, {abnormal_count} abnormal")
            return results

        except Exception as e:
            current_app.logger.error(f"Error analyzing file {csv_path}: {str(e)}")
            raise

    def create_results_csv(self, results, output_path):
        """
        Create CSV file with results

        Args:
            results: Dict containing analysis results
            output_path: Path where to save the CSV

        Returns:
            str: Path to the created CSV file
        """
        try:
            output_df = pd.DataFrame({
                'beat_id': range(results['total_beats']),
                'prediction': results['predictions'],
                'confidence': results['confidences']
            })

            output_df.to_csv(output_path, index=False)
            current_app.logger.info(f"Results CSV saved to {output_path}")
            return output_path

        except Exception as e:
            current_app.logger.error(f"Error creating results CSV: {str(e)}")
            raise

    @staticmethod
    def cleanup_graphs(analysis_id):
        """
        Clean up graph files for a specific analysis

        Args:
            analysis_id: ID of the analysis to clean up
        """
        try:
            graphs_folder = current_app.config['GRAPHS_FOLDER']
            pattern = f"analysis_{analysis_id}_beat_*.png"

            for filename in os.listdir(graphs_folder):
                if filename.startswith(f"analysis_{analysis_id}_"):
                    file_path = os.path.join(graphs_folder, filename)
                    os.remove(file_path)

            current_app.logger.info(f"Cleaned up graphs for analysis {analysis_id}")

        except Exception as e:
            current_app.logger.error(f"Error cleaning up graphs: {str(e)}")
