"""
MIT License

Copyright (c) 2025 Ilker M. Karakas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Optional web dashboard for the Military Leadership Assessment system.
Provides interactive visualizations and real-time predictions through a web interface.
Can be run independently without affecting CLI functionality.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
from pathlib import Path
import logging

from .predictor import predict_competencies
from .visualizations import (
    plot_prediction_radar,
    plot_model_performance,
    plot_feature_importance
)
from .data_loader import load_data_from_jsonl

logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self, model_path="models/leadership_model.joblib", 
                 preprocessor_path="models/preprocessor.joblib",
                 data_path="data/synthetic_data/generated_1000.jsonl"):
        """Initialize the dashboard with model and data paths."""
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.data_path = data_path
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Create necessary directories
        self.static_dir = Path("web/static")
        self.templates_dir = Path("web/templates")
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load initial data
        try:
            self.df = load_data_from_jsonl(data_path)
            logger.info(f"Loaded {len(self.df)} records from {data_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    def setup_routes(self):
        """Set up Flask routes."""
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                officer_data = request.json
                predictions = predict_competencies(
                    officer_data,
                    model_path=self.model_path,
                    preprocessor_path=self.preprocessor_path
                )
                
                # Generate radar plot
                plot_path = self.static_dir / 'img' / 'prediction_radar.png'
                plot_path.parent.mkdir(exist_ok=True)
                plot_prediction_radar(predictions, save_path=str(plot_path))
                
                return jsonify({
                    'success': True,
                    'predictions': predictions,
                    'plot_url': '/static/img/prediction_radar.png'
                })
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 400

        @self.app.route('/analytics')
        def analytics():
            return render_template('analytics.html')

        @self.app.route('/api/stats')
        def get_stats():
            try:
                stats = {
                    'total_officers': len(self.df),
                    'branches': self.df['branch'].value_counts().to_dict(),
                    'ranks': self.df['rank'].value_counts().to_dict(),
                    'avg_competencies': self.df['leadership_competency_summary'].apply(
                        lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
                    ).mean().to_dict()
                }
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/statistics')
        def api_statistics():
            try:
                df = self.df.copy()
                # Flatten the 'competencies' column if it exists
                if 'competencies' in df.columns:
                    comp_df = pd.json_normalize(df['competencies'])
                    comp_df.columns = [str(col) for col in comp_df.columns]
                    df = pd.concat([df.drop('competencies', axis=1), comp_df], axis=1)
                competency_cols = [
                    "thinks_strategically", "possesses_english_language_skills", "engages_in_ethical_reasoning",
                    "builds_trust", "facilitates_collaboration_communication", "builds_consensus",
                    "integrates_technology", "understands_effects_of_leveraging_technology",
                    "understands_capabilities", "instills_need_for_change", "anticipates_change_requirements",
                    "provides_support_for_change", "enables_empowers_others", "upholds_principles",
                    "relationship_oriented", "thrives_in_ambiguity", "demonstrates_resilience",
                    "learning_oriented", "operates_in_nato_context", "operates_in_military_context",
                    "operates_in_cross_cultural_context"
                ]
                # Stats
                stats = {
                    'total_officers': int(len(df)),
                    'avg_competency': float(df[competency_cols].mean().mean()),
                    'highest_competency': float(df[competency_cols].max().max()),
                    'model_accuracy': 0.85  # Placeholder, replace with real value if available
                }
                # Distribution of a sample competency
                distribution = df['thinks_strategically'].value_counts().sort_index().to_dict()
                # Rank comparison
                rank_comparison = []
                for rank, group in df.groupby('rank'):
                    rank_comparison.append({
                        'rank': rank,
                        'competencies': group[competency_cols].mean().to_dict(),
                        'color': 'rgba(54, 162, 235, 1)'
                    })
                # Branch comparison
                branch_comparison = df.groupby('branch')["thinks_strategically"].mean().to_dict()
                return jsonify({
                    'success': True,
                    'stats': stats,
                    'distribution': distribution,
                    'rank_comparison': rank_comparison,
                    'branch_comparison': branch_comparison
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def run_dashboard():
    """Entry point for running the dashboard."""
    dashboard = Dashboard()
    dashboard.run()

if __name__ == '__main__':
    run_dashboard() 