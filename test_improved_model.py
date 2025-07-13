#!/usr/bin/env python3
"""
Test script to evaluate the improved model's ability to generate
empathetic, actionable advice instead of generic responses.
"""

import pandas as pd
import torch
from src.get_text import GenerateText
import re
import json
from datetime import datetime

class ImprovedModelTester:
    def __init__(self):
        """Initialize the tester with the trained model."""
        try:
            self.generator = GenerateText()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.generator = None
    
    def create_test_scenarios(self):
        """Create diverse test scenarios for evaluation."""
        return [
            {
                "name": "Major Depression - High Severity",
                "input": "I'm completely overwhelmed by this emptiness inside and can't seem to get out of bed most days",
                "patient_data": {
                    "Diagnosis": "Major Depressive Disorder",
                    "Symptom Severity (1-10)": 9,
                    "Mood Score (1-10)": 2,
                    "Sleep Quality (1-10)": 3,
                    "Therapy Type": "Cognitive Behavioral Therapy",
                    "Adherence to Treatment (%)": 65
                },
                "expected_elements": ["empathy", "CBT techniques", "behavioral activation", "sleep"]
            },
            {
                "name": "Panic Disorder - Acute Episode",
                "input": "I'm having these terrifying panic attacks and feel like I'm going to die",
                "patient_data": {
                    "Diagnosis": "Panic Disorder",
                    "Symptom Severity (1-10)": 8,
                    "Mood Score (1-10)": 4,
                    "Sleep Quality (1-10)": 5,
                    "Therapy Type": "Exposure Therapy",
                    "Adherence to Treatment (%)": 80
                },
                "expected_elements": ["grounding techniques", "breathing exercises", "exposure", "reassurance"]
            },
            {
                "name": "Generalized Anxiety - Treatment Adherence Issues",
                "input": "I'm struggling to keep up with my treatment plan and the anxiety is getting worse",
                "patient_data": {
                    "Diagnosis": "Generalized Anxiety Disorder",
                    "Symptom Severity (1-10)": 7,
                    "Mood Score (1-10)": 5,
                    "Sleep Quality (1-10)": 6,
                    "Therapy Type": "Mindfulness-Based Therapy",
                    "Adherence to Treatment (%)": 45
                },
                "expected_elements": ["mindfulness", "adherence support", "treatment consistency", "progress"]
            },
            {
                "name": "Depression - Recovery Progress",
                "input": "I've been feeling a bit better lately but still have bad days",
                "patient_data": {
                    "Diagnosis": "Major Depressive Disorder",
                    "Symptom Severity (1-10)": 4,
                    "Mood Score (1-10)": 6,
                    "Sleep Quality (1-10)": 7,
                    "Therapy Type": "Interpersonal Therapy",
                    "Adherence to Treatment (%)": 85
                },
                "expected_elements": ["progress acknowledgment", "relapse prevention", "continued engagement"]
            }
        ]
    
    def analyze_response_quality(self, response, expected_elements):
        """Analyze the quality of a generated response."""
        metrics = {
            "has_empathy": False,
            "has_advice": False,
            "is_specific": False,
            "acknowledges_progress": False,
            "therapy_specific": False,
            "actionable_count": 0,
            "question_count": 0,
            "length": len(response.split())
        }
        
        response_lower = response.lower()
        
        # Check for empathy
        empathy_indicators = [
            "i hear", "i understand", "it's clear", "your struggle", 
            "valid", "support you", "courage", "difficult", "challenging"
        ]
        metrics["has_empathy"] = any(indicator in response_lower for indicator in empathy_indicators)
        
        # Check for actionable advice
        advice_indicators = [
            "try", "practice", "use", "consider", "start", "focus", 
            "develop", "build", "continue", "remember", "apply"
        ]
        metrics["actionable_count"] = sum(1 for indicator in advice_indicators if indicator in response_lower)
        metrics["has_advice"] = metrics["actionable_count"] > 0
        
        # Count questions (should be minimal)
        metrics["question_count"] = response.count("?")
        
        # Check for therapy-specific content
        therapy_terms = [
            "cbt", "cognitive", "behavioral", "mindfulness", "exposure", 
            "interpersonal", "dbt", "breathing", "grounding", "thought"
        ]
        metrics["therapy_specific"] = any(term in response_lower for term in therapy_terms)
        
        # Check for specific techniques vs generic advice
        specific_techniques = [
            "breathing", "meditation", "exercise", "sleep", "medication",
            "thought record", "grounding", "exposure", "relaxation"
        ]
        metrics["is_specific"] = any(technique in response_lower for technique in specific_techniques)
        
        # Check for progress acknowledgment
        progress_terms = ["progress", "improved", "better", "recovery", "healing", "journey"]
        metrics["acknowledges_progress"] = any(term in response_lower for term in progress_terms)
        
        return metrics
    
    def run_comprehensive_test(self):
        """Run comprehensive testing of the improved model."""
        if not self.generator:
            print("❌ Cannot run tests - model not loaded")
            return
        
        print("🧪 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        test_scenarios = self.create_test_scenarios()
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Test {i}: {scenario['name']}")
            print(f"Input: {scenario['input']}")
            
            try:
                # Generate response
                response = self.generator.generate(
                    scenario['input'], 
                    scenario['patient_data']
                )
                print(f"Response: {response}")
                
                # Analyze quality
                metrics = self.analyze_response_quality(response, scenario['expected_elements'])
                
                # Store results
                result = {
                    "scenario": scenario['name'],
                    "input": scenario['input'],
                    "response": response,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                
                # Print analysis
                print(f"\n📊 Analysis:")
                print(f"   ✅ Empathetic: {metrics['has_empathy']}")
                print(f"   ✅ Contains Advice: {metrics['has_advice']} ({metrics['actionable_count']} actionable items)")
                print(f"   ✅ Therapy-Specific: {metrics['therapy_specific']}")
                print(f"   ✅ Specific Techniques: {metrics['is_specific']}")
                print(f"   ✅ Questions (should be low): {metrics['question_count']}")
                print(f"   📏 Response Length: {metrics['length']} words")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                results.append({
                    "scenario": scenario['name'],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"tests/improved_model_evaluation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Summary statistics
        successful_tests = [r for r in results if 'error' not in r]
        if successful_tests:
            empathy_rate = sum(1 for r in successful_tests if r['metrics']['has_empathy']) / len(successful_tests)
            advice_rate = sum(1 for r in successful_tests if r['metrics']['has_advice']) / len(successful_tests)
            therapy_rate = sum(1 for r in successful_tests if r['metrics']['therapy_specific']) / len(successful_tests)
            avg_advice_count = sum(r['metrics']['actionable_count'] for r in successful_tests) / len(successful_tests)
            avg_questions = sum(r['metrics']['question_count'] for r in successful_tests) / len(successful_tests)
            
            print(f"\n📈 Summary Statistics:")
            print(f"   🤝 Empathy Rate: {empathy_rate:.1%}")
            print(f"   💡 Advice Rate: {advice_rate:.1%}")
            print(f"   🎯 Therapy-Specific Rate: {therapy_rate:.1%}")
            print(f"   📋 Avg Actionable Items: {avg_advice_count:.1f}")
            print(f"   ❓ Avg Questions: {avg_questions:.1f}")
        
        return results

if __name__ == "__main__":
    tester = ImprovedModelTester()
    tester.run_comprehensive_test()
