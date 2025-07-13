#!/usr/bin/env python3
"""
Improved Text Generation Testing Suite for Alita Medical AI
Testing with more directive prompts to generate actionable advice
"""

import sys
import os
import logging
from datetime import datetime
import torch
import json
import time
from typing import Dict, List, Any

# Add project root to Python path
sys.path.append('/Users/jaceysimpson/Vscode/Alita')

from src.get_text import GenerateText

class ImprovedTextGenerationTester:
    def __init__(self, model_path="model/medical_t5_model.pth", log_to_file=True):
        """Initialize the improved text generation tester."""
        self.model_path = model_path
        self.setup_logging(log_to_file)
        self.generator = None
        self.test_results = []
        
    def setup_logging(self, log_to_file=True):
        """Setup logging to both console and file."""
        os.makedirs('tests', exist_ok=True)
        os.makedirs('tests/logs', exist_ok=True)
        
        self.logger = logging.getLogger('ImprovedTextGenerationTester')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(f'tests/logs/improved_generation_test_{timestamp}.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info("=" * 60)
        self.logger.info("IMPROVED TEXT GENERATION TESTER INITIALIZED")
        self.logger.info("=" * 60)

    def initialize_generator(self):
        """Initialize the GenerateText model."""
        self.logger.info("🤖 Initializing GenerateText model...")
        
        try:
            self.generator = GenerateText(model_path=self.model_path)
            self.logger.info("✅ GenerateText model initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing GenerateText: {str(e)}")
            raise

    def get_improved_test_scenarios(self) -> List[Dict[str, Any]]:
        """Define improved test scenarios with more directive inputs."""
        return [
            {
                "name": "Depression - High Severity - Specific Request",
                "input_text": "I've been on Sertraline for 12 weeks but still feel severely depressed with poor sleep. What specific coping strategies can help me right now?",
                "parameters": {
                    "Age": 35,
                    "Gender": "Female",
                    "Diagnosis": "Major Depressive Disorder",
                    "Symptom Severity (1-10)": 8,
                    "Mood Score (1-10)": 3,
                    "Sleep Quality (1-10)": 2,
                    "Physical Activity (hrs/week)": 1,
                    "Medication": "Sertraline",
                    "Therapy Type": "CBT",
                    "Treatment Start Date": "2024-01-15",
                    "Treatment Duration (weeks)": 12,
                    "Stress Level (1-10)": 9,
                    "Outcome": "Improving",
                    "Treatment Progress (1-10)": 5,
                    "AI-Detected Emotional State": "Severely Depressed",
                    "Adherence to Treatment (%)": 80
                },
                "expected_themes": ["sleep hygiene", "exercise", "therapy techniques", "medication adherence", "daily routine"]
            },
            {
                "name": "Anxiety - Practical Coping Request",
                "input_text": "I'm having panic attacks at work and my Alprazolam isn't always available. What are some immediate techniques I can use to calm down?",
                "parameters": {
                    "Age": 28,
                    "Gender": "Male",
                    "Diagnosis": "Generalized Anxiety Disorder",
                    "Symptom Severity (1-10)": 6,
                    "Mood Score (1-10)": 5,
                    "Sleep Quality (1-10)": 4,
                    "Physical Activity (hrs/week)": 3,
                    "Medication": "Alprazolam",
                    "Therapy Type": "Exposure Therapy",
                    "Treatment Start Date": "2024-02-01",
                    "Treatment Duration (weeks)": 8,
                    "Stress Level (1-10)": 7,
                    "Outcome": "Stable",
                    "Treatment Progress (1-10)": 6,
                    "AI-Detected Emotional State": "Anxious",
                    "Adherence to Treatment (%)": 90
                },
                "expected_themes": ["breathing exercises", "grounding techniques", "mindfulness", "workplace strategies", "emergency coping"]
            },
            {
                "name": "Sleep Issues - Direct Help Request",
                "input_text": "I'm 68 years old and taking Melatonin but still can't sleep. What daily habits should I change to improve my sleep quality?",
                "parameters": {
                    "Age": 68,
                    "Gender": "Female",
                    "Diagnosis": "Insomnia",
                    "Symptom Severity (1-10)": 7,
                    "Mood Score (1-10)": 6,
                    "Sleep Quality (1-10)": 2,
                    "Physical Activity (hrs/week)": 2,
                    "Medication": "Melatonin",
                    "Therapy Type": "Sleep Hygiene Counseling",
                    "Treatment Start Date": "2024-03-01",
                    "Treatment Duration (weeks)": 6,
                    "Stress Level (1-10)": 5,
                    "Outcome": "In Progress",
                    "Treatment Progress (1-10)": 4,
                    "AI-Detected Emotional State": "Tired",
                    "Adherence to Treatment (%)": 95
                },
                "expected_themes": ["bedtime routine", "screen time", "exercise timing", "caffeine", "room environment"]
            },
            {
                "name": "PTSD - Coping Strategy Request",
                "input_text": "My EMDR therapy is helping but I still get triggered by flashbacks. What can I do when I feel a flashback starting?",
                "parameters": {
                    "Age": 24,
                    "Gender": "Male",
                    "Diagnosis": "PTSD",
                    "Symptom Severity (1-10)": 9,
                    "Mood Score (1-10)": 3,
                    "Sleep Quality (1-10)": 2,
                    "Physical Activity (hrs/week)": 1,
                    "Medication": "Prazosin",
                    "Therapy Type": "EMDR",
                    "Treatment Start Date": "2024-01-01",
                    "Treatment Duration (weeks)": 16,
                    "Stress Level (1-10)": 9,
                    "Outcome": "Improving",
                    "Treatment Progress (1-10)": 5,
                    "AI-Detected Emotional State": "Traumatized",
                    "Adherence to Treatment (%)": 85
                },
                "expected_themes": ["grounding techniques", "breathing", "safe space", "EMDR tools", "support system"]
            },
            {
                "name": "Depression - Exercise Motivation",
                "input_text": "I know exercise helps depression but I have no energy or motivation. How can I start moving when everything feels impossible?",
                "parameters": {
                    "Age": 29,
                    "Gender": "Male",
                    "Diagnosis": "Major Depressive Disorder",
                    "Symptom Severity (1-10)": 7,
                    "Mood Score (1-10)": 3,
                    "Sleep Quality (1-10)": 4,
                    "Physical Activity (hrs/week)": 0,
                    "Medication": "Fluoxetine",
                    "Therapy Type": "CBT",
                    "Treatment Start Date": "2024-02-01",
                    "Treatment Duration (weeks)": 10,
                    "Stress Level (1-10)": 6,
                    "Outcome": "Improving",
                    "Treatment Progress (1-10)": 4,
                    "AI-Detected Emotional State": "Lethargic",
                    "Adherence to Treatment (%)": 85
                },
                "expected_themes": ["small steps", "gentle exercise", "routine building", "motivation", "energy"]
            },
            {
                "name": "Anxiety - Social Situations",
                "input_text": "I avoid social situations because of my anxiety, but I'm lonely. What are some gradual steps to become more social again?",
                "parameters": {
                    "Age": 26,
                    "Gender": "Female",
                    "Diagnosis": "Social Anxiety Disorder",
                    "Symptom Severity (1-10)": 8,
                    "Mood Score (1-10)": 4,
                    "Sleep Quality (1-10)": 5,
                    "Physical Activity (hrs/week)": 2,
                    "Medication": "Sertraline",
                    "Therapy Type": "Exposure Therapy",
                    "Treatment Start Date": "2024-01-01",
                    "Treatment Duration (weeks)": 12,
                    "Stress Level (1-10)": 8,
                    "Outcome": "Improving",
                    "Treatment Progress (1-10)": 5,
                    "AI-Detected Emotional State": "Anxious",
                    "Adherence to Treatment (%)": 90
                },
                "expected_themes": ["gradual exposure", "comfort zone", "small groups", "online first", "support"]
            }
        ]

    def analyze_advice_quality(self, response: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of advice vs questions in the response."""
        if not response or len(response.strip()) == 0:
            return {"score": 0.0, "advice_count": 0, "question_count": 0, "advice_ratio": 0.0}
        
        # Count advice statements vs questions
        sentences = response.split('.')
        questions = response.count('?')
        advice_indicators = ["try", "practice", "consider", "start", "avoid", "limit", "increase", "decrease", "establish", "maintain", "focus on", "remember to"]
        
        advice_count = 0
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in advice_indicators):
                advice_count += 1
        
        total_sentences = len([s for s in sentences if s.strip()])
        question_count = questions
        
        advice_ratio = advice_count / max(total_sentences, 1)
        question_ratio = question_count / max(total_sentences, 1)
        
        # Calculate quality score
        score = 0.0
        max_score = 10.0
        
        # Length appropriateness (2 points)
        if 30 <= len(response) <= 300:
            score += 2.0
        elif len(response) > 15:
            score += 1.0
        
        # Advice vs questions ratio (4 points)
        if advice_ratio > question_ratio:
            score += 4.0
        elif advice_ratio > 0:
            score += 2.0
        
        # Medical appropriateness (2 points)
        medical_terms = ["therapy", "treatment", "medication", "professional", "counselor", "doctor"]
        if any(term in response.lower() for term in medical_terms):
            score += 2.0
        
        # Specific actionable advice (2 points)
        actionable_words = ["practice", "try", "start with", "begin by", "focus on", "limit", "avoid", "establish"]
        actionable_count = sum(1 for word in actionable_words if word in response.lower())
        score += min(2.0, actionable_count * 0.5)
        
        return {
            "score": min(score, max_score),
            "advice_count": advice_count,
            "question_count": question_count,
            "advice_ratio": advice_ratio,
            "question_ratio": question_ratio,
            "actionable_count": actionable_count
        }

    def test_single_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single scenario and return detailed results."""
        self.logger.info(f"🧪 Testing scenario: {scenario['name']}")
        self.logger.info(f"💬 Input: {scenario['input_text']}")
        self.logger.info(f"👤 Patient: {scenario['parameters']['Age']}y {scenario['parameters']['Gender']}, {scenario['parameters']['Diagnosis']}")
        
        start_time = time.time()
        
        try:
            response = self.generator.generate(
                input_text=scenario['input_text'],
                exclusive_parameters=scenario['parameters']
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if response is None or "Error" in str(response):
                self.logger.error(f"❌ {scenario['name']}: Failed to generate response")
                return {
                    "scenario": scenario['name'],
                    "success": False,
                    "response": response,
                    "generation_time": generation_time,
                    "error": "Failed to generate valid response"
                }
            
            self.logger.info(f"🤖 Generated response: {response}")
            self.logger.info(f"⏱️ Generation time: {generation_time:.2f} seconds")
            
            # Analyze advice quality
            analysis = self.analyze_advice_quality(response, scenario)
            
            # Log analysis details
            self.logger.info(f"📊 Analysis:")
            self.logger.info(f"   - Advice statements: {analysis['advice_count']}")
            self.logger.info(f"   - Questions: {analysis['question_count']}")
            self.logger.info(f"   - Advice ratio: {analysis['advice_ratio']:.2f}")
            self.logger.info(f"   - Quality score: {analysis['score']:.1f}/10")
            
            result = {
                "scenario": scenario['name'],
                "success": True,
                "input": scenario['input_text'],
                "response": response,
                "generation_time": generation_time,
                "analysis": analysis,
                "patient_profile": {
                    "age": scenario['parameters']['Age'],
                    "gender": scenario['parameters']['Gender'],
                    "diagnosis": scenario['parameters']['Diagnosis'],
                    "severity": scenario['parameters']['Symptom Severity (1-10)']
                }
            }
            
            self.logger.info(f"✅ {scenario['name']}: Success")
            return result
            
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            
            self.logger.error(f"❌ {scenario['name']}: Error - {str(e)}")
            return {
                "scenario": scenario['name'],
                "success": False,
                "response": None,
                "generation_time": generation_time,
                "error": str(e)
            }

    def run_improved_test(self):
        """Run improved text generation tests."""
        self.logger.info("🚀 Starting improved text generation testing...")
        
        try:
            self.initialize_generator()
            scenarios = self.get_improved_test_scenarios()
            self.logger.info(f"📋 Testing {len(scenarios)} improved scenarios...")
            
            for i, scenario in enumerate(scenarios, 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"TEST {i}/{len(scenarios)}: {scenario['name']}")
                self.logger.info(f"{'='*60}")
                
                result = self.test_single_scenario(scenario)
                self.test_results.append(result)
                
                time.sleep(1)
            
            self.generate_improved_summary()
            self.logger.info("🎉 ALL IMPROVED TESTS COMPLETED!")
            
        except Exception as e:
            self.logger.error(f"💥 TESTING FAILED: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def generate_improved_summary(self):
        """Generate and save improved test summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 IMPROVED TEST SUMMARY")
        self.logger.info("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result['success'])
        
        if successful_tests > 0:
            avg_generation_time = sum(result['generation_time'] for result in self.test_results if result['success']) / successful_tests
            avg_quality_score = sum(result['analysis']['score'] for result in self.test_results if result['success']) / successful_tests
            avg_advice_ratio = sum(result['analysis']['advice_ratio'] for result in self.test_results if result['success']) / successful_tests
            avg_question_ratio = sum(result['analysis']['question_ratio'] for result in self.test_results if result['success']) / successful_tests
        else:
            avg_generation_time = avg_quality_score = avg_advice_ratio = avg_question_ratio = 0
        
        self.logger.info(f"🎯 Overall Results:")
        self.logger.info(f"   - Total tests: {total_tests}")
        self.logger.info(f"   - Successful: {successful_tests}")
        self.logger.info(f"   - Success rate: {(successful_tests/total_tests)*100:.1f}%")
        self.logger.info(f"   - Average generation time: {avg_generation_time:.2f} seconds")
        self.logger.info(f"   - Average quality score: {avg_quality_score:.1f}/10")
        self.logger.info(f"   - Average advice ratio: {avg_advice_ratio:.2f}")
        self.logger.info(f"   - Average question ratio: {avg_question_ratio:.2f}")
        
        # Individual results
        self.logger.info(f"\n📋 Individual Test Results:")
        for result in self.test_results:
            if result['success']:
                analysis = result['analysis']
                self.logger.info(f"   ✅ {result['scenario']}:")
                self.logger.info(f"      Quality: {analysis['score']:.1f}/10, Advice: {analysis['advice_count']}, Questions: {analysis['question_count']}")
            else:
                self.logger.info(f"   ❌ {result['scenario']}: {result.get('error', 'Unknown error')}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'tests/improved_generation_results_{timestamp}.json'
        
        summary_data = {
            "timestamp": timestamp,
            "test_type": "improved_generation",
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (successful_tests/total_tests)*100,
            "average_generation_time": avg_generation_time,
            "average_quality_score": avg_quality_score,
            "average_advice_ratio": avg_advice_ratio,
            "average_question_ratio": avg_question_ratio,
            "detailed_results": self.test_results
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {str(e)}")

def main():
    """Main function to run the improved text generation tester."""
    print("🧪 Improved Text Generation Tester Starting...")
    print("📋 Testing with more directive prompts for better advice")
    print("🎯 Analyzing advice vs questions ratio")
    print("=" * 60)
    
    tester = ImprovedTextGenerationTester(
        model_path="model/medical_t5_model.pth",
        log_to_file=True
    )
    
    try:
        tester.run_improved_test()
        print("\n🎉 All improved tests completed!")
        print("📁 Check tests/logs/ for detailed logs")
        print("📊 Check tests/ for JSON results")
        
    except Exception as e:
        print(f"💥 Testing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()