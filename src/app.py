import os
import sqlite3
import pickle
import joblib
import requests
import pandas as pd
import numpy as np
import json
import io
import logging
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, session, send_file
from flask_cors import CORS
from flask_session import Session
import warnings
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agriculture.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'smart-agriculture-ai-platform-secret-key-2026')
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_FILE_DIR = './flask_session'
    DATABASE = 'agriculture.db'
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')  # Add your API key here
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
    ENCODER_PATH = os.path.join(BASE_DIR, 'target_encoder.pkl')
    FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
    
    # Constants with value ranges
    SOIL_TYPES = ['Sandy', 'Loamy', 'Clayey', 'Black', 'Red']
    CROP_TYPES = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Pulses', 'Ground Nuts', 'Millets']
    
    # Value ranges for validation
    TEMP_MIN = -10
    TEMP_MAX = 50
    HUMIDITY_MIN = 0
    HUMIDITY_MAX = 100
    MOISTURE_MIN = 0
    MOISTURE_MAX = 100
    NUTRIENT_MIN = 0
    NUTRIENT_MAX = 300

app = Flask(__name__)
app.config.from_object(Config)

# Ensure session directory exists
os.makedirs(Config.SESSION_FILE_DIR, exist_ok=True)
os.makedirs(Config.FONTS_DIR, exist_ok=True)

Session(app)
CORS(app)

# ============================================================================
# TRANSLATIONS
# ============================================================================

TRANSLATIONS = {
    'en': {
        # UI Elements
        'app_name': 'Smart Agriculture AI Platform',
        'tagline': 'AI-Powered Precision Farming Assistant',
        'home': 'Home',
        'new_recommendation': 'New Recommendation',
        'weather': 'Weather',
        'history': 'History',
        'analytics': 'Analytics',
        
        # Form Labels
        'input_parameters': 'Input Parameters',
        'auto_fetch': 'Auto-fetch weather',
        'manual_entry': 'Manual Entry',
        'city_input': 'Enter city name',
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'moisture': 'Soil Moisture (%)',
        'soil_type': 'Soil Type',
        'crop_type': 'Crop Type',
        'nitrogen': 'Nitrogen (N) - kg/ha',
        'phosphorous': 'Phosphorous (P) - kg/ha',
        'potassium': 'Potassium (K) - kg/ha',
        'get_recommendation': 'Get AI Recommendation',
        
        # Soil Types
        'sandy': 'Sandy',
        'loamy': 'Loamy',
        'clayey': 'Clayey',
        'black': 'Black',
        'red': 'Red',
        
        # Crop Types
        'rice': 'Rice',
        'wheat': 'Wheat',
        'maize': 'Maize',
        'cotton': 'Cotton',
        'sugarcane': 'Sugarcane',
        'pulses': 'Pulses',
        'ground_nuts': 'Ground Nuts',
        'millets': 'Millets',
        
        # Result Labels
        'no_fertilizer_needed': 'No Additional Fertilizer Required',
        'dose_per_acre': 'kg per acre',
        'confidence': 'Confidence',
        'deficiency_analysis': 'Nutrient Deficiency Analysis',
        'nitrogen_deficit': 'Nitrogen',
        'phosphorous_deficit': 'Phosphorous',
        'potassium_deficit': 'Potassium',
        'requirement_fulfilled': 'Requirement Fulfilled',
        'download_report': 'Download Report',
        'speak_result': 'Listen',
        
        # Messages
        'fetching_location': 'Fetching location...',
        'fetching_weather': 'Fetching weather data...',
        'location_error': 'Unable to get location',
        'weather_error': 'Weather fetch failed',
        'weather_fetched': 'Weather data fetched successfully!',
        'use_my_location': 'Use My Location',
        'analyzing': 'Analyzing your soil and crop data...',
        
        # Weather Labels
        'feels_like': 'Feels like',
        'kmh': 'km/h',
        'air_quality_good': 'Good',
        'air_quality_moderate': 'Moderate',
        'air_quality_poor': 'Poor',
        'humidity_label': 'Humidity',
        'wind_label': 'Wind',
        'uv_label': 'UV Index',
        'air_quality_label': 'Air Quality',
        'rain_label': 'Rain',
        'sun_label': 'Sunrise/Sunset',
        'hourly_forecast': 'Hourly Forecast',
        'weekly_forecast': '7-Day Forecast',
        
        # History & Analytics
        'history_title': 'Prediction History',
        'all_crops': 'All Crops',
        'download_pdf': 'Download PDF',
        'date': 'Date',
        'fertilizer': 'Fertilizer',
        'dose': 'Dose (kg)',
        'most_recommended': 'Most Recommended',
        'avg_nitrogen': 'Avg Nitrogen',
        'avg_phosphorous': 'Avg Phosphorous',
        'avg_potassium': 'Avg Potassium',
        'avg_confidence': 'Avg Confidence',
        'avg_dose': 'Avg Dose',
        'most_grown_crop': 'Most Grown Crop',
        'total_predictions': 'Total Predictions',
        'predictions': 'Predictions',
        'fertilizer_distribution': 'Fertilizer Distribution',
        'crop_distribution': 'Crop Distribution',
        'monthly_trends': 'Monthly Prediction Trends',
        'confidence_trend': 'Confidence Trend',
        'nutrient_comparison': 'Nutrient Averages Comparison',
        'total_predictions_label': 'Total Predictions',
        'most_recommended_label': 'Most Recommended',
        'most_grown_label': 'Most Grown Crop',
        'avg_nitrogen_label': 'Avg Nitrogen',
        'avg_confidence_label': 'Avg Confidence',
        'avg_dose_label': 'Avg Dose',
        
        # Welcome
        'welcome_title': 'Welcome to Smart Agriculture AI Platform',
        'welcome_text': 'Get personalized fertilizer recommendations powered by AI and real-time weather data',
        'get_started': 'Get Started',
        
        # AI Assistant
        'ai_greeting': 'Hello! I am your AI farming assistant. I will help you fill the form. You have 30 seconds to respond.',
        'ask_weather': 'Do you want auto-fetch weather? Say yes or no.',
        'ask_temperature': 'What is the temperature in Celsius?',
        'ask_humidity': 'What is the humidity percentage?',
        'ask_moisture': 'What is the soil moisture percentage?',
        'ask_soil': 'What is your soil type? Sandy, Loamy, Clayey, Black, or Red?',
        'ask_crop': 'What crop are you growing? Rice, Wheat, Maize, Cotton, Sugarcane, Pulses, Ground Nuts, or Millets?',
        'ask_nitrogen': 'What is your soil nitrogen level in kg per hectare?',
        'ask_phosphorous': 'What is your soil phosphorous level in kg per hectare?',
        'ask_potassium': 'What is your soil potassium level in kg per hectare?',
        'ai_thanks': 'Thank you! Processing your information...',
        'ai_no_response': 'No response detected. Please try again.',
        'ai_stopping': 'No response. Stopping AI assistant.',
        'ai_complete': 'All questions completed! Getting your recommendation...',
        'ai_error': 'Sorry, I did not understand. Please try again.',
        'listening': 'Listening...',
        'speaking': 'Speaking...',
        'processing': 'Processing...',
        'repeat': 'Repeat',
        'stop': 'Stop',
        'yes_detected': 'Yes detected',
        'no_detected': 'No detected',
        'please_speak': 'Please speak your answer',
        'time_remaining': 'Time remaining',
        'seconds': 'seconds',
        'recommended_fertilizer': 'Recommended fertilizer',
        
        # PDF Report
        'report_title': 'Fertilizer Recommendation Report',
        'report_subtitle': 'AI-Powered Precision Farming',
        'generated_on': 'Generated on',
        'location': 'Location',
        'weather_conditions': 'Weather Conditions',
        'soil_analysis': 'Soil Analysis',
        'current': 'Current',
        'required': 'Required',
        'ai_recommendation': 'AI Recommendation',
        'explanation': 'Explanation',
        'application_instructions': 'Application Instructions',
        'safety_precautions': 'Safety Precautions',
        'irrigation_advice': 'Irrigation Advice',
        'application_timing': 'Application Timing',
        'application_method': 'Application Method',
        'storage_instructions': 'Storage Instructions',
        'footer_text': 'Smart Agriculture AI Platform - Precision Farming for a Sustainable Future',
        
        # Validation Errors
        'error_temperature_required': 'Temperature is required',
        'error_temperature_range': f'Temperature must be between {Config.TEMP_MIN}°C and {Config.TEMP_MAX}°C',
        'error_humidity_required': 'Humidity is required',
        'error_humidity_range': f'Humidity must be between {Config.HUMIDITY_MIN}% and {Config.HUMIDITY_MAX}%',
        'error_moisture_required': 'Soil moisture is required',
        'error_moisture_range': f'Soil moisture must be between {Config.MOISTURE_MIN}% and {Config.MOISTURE_MAX}%',
        'error_soil_required': 'Soil type is required',
        'error_crop_required': 'Crop type is required',
        'error_nitrogen_required': 'Nitrogen level is required',
        'error_nitrogen_range': f'Nitrogen must be between {Config.NUTRIENT_MIN} and {Config.NUTRIENT_MAX} kg/ha',
        'error_phosphorous_required': 'Phosphorous level is required',
        'error_phosphorous_range': f'Phosphorous must be between {Config.NUTRIENT_MIN} and {Config.NUTRIENT_MAX} kg/ha',
        'error_potassium_required': 'Potassium level is required',
        'error_potassium_range': f'Potassium must be between {Config.NUTRIENT_MIN} and {Config.NUTRIENT_MAX} kg/ha',
        'error_invalid_soil': f'Invalid soil type. Choose from: {", ".join(Config.SOIL_TYPES)}',
        'error_invalid_crop': f'Invalid crop type. Choose from: {", ".join(Config.CROP_TYPES)}',
        'error_missing_fields': 'Missing required fields',
        'error_prediction_service': 'Prediction service error',
        'error_weather_service': 'Weather service error',
        'error_network': 'Network error. Please try again.',
        
        # Optimization Labels
        'optimization_title': 'Optimized Fertilizer Schedule',
        'stage': 'Stage',
        'purpose': 'Purpose',
        'time': 'Time',
        'total_nutrient_supply': 'Total Nutrient Supply',
        'optimization_score': 'Optimization Score',
        'soil_nutrient_balance': 'Soil Nutrient Balance',
        'optimal': 'Optimal',
        'irrigation_recommendation': 'Irrigation Recommendation',
        'summary': 'Summary',
        'stage_1': 'Basal Application',
        'stage_2': 'Vegetative Stage',
        'stage_3': 'Flowering Stage',
        'purpose_1': 'Root development and early growth',
        'purpose_2': 'Vegetative growth and tillering',
        'purpose_3': 'Flowering and fruit development'
    },
    'hi': {
        # Hindi translations (keep as in original)
        'app_name': 'स्मार्ट कृषि AI प्लेटफॉर्म',
        'tagline': 'AI-संचालित सटीक कृषि सहायक',
        'home': 'होम',
        'new_recommendation': 'नई सिफारिश',
        'weather': 'मौसम',
        'history': 'इतिहास',
        'analytics': 'विश्लेषण',
        'input_parameters': 'इनपुट पैरामीटर',
        'auto_fetch': 'स्वचालित मौसम',
        'manual_entry': 'मैन्युअल प्रविष्टि',
        'city_input': 'शहर का नाम दर्ज करें',
        'temperature': 'तापमान (°C)',
        'humidity': 'आर्द्रता (%)',
        'moisture': 'मिट्टी की नमी (%)',
        'soil_type': 'मिट्टी का प्रकार',
        'crop_type': 'फसल का प्रकार',
        'nitrogen': 'नाइट्रोजन (N) - किग्रा/हेक्टेयर',
        'phosphorous': 'फास्फोरस (P) - किग्रा/हेक्टेयर',
        'potassium': 'पोटेशियम (K) - किग्रा/हेक्टेयर',
        'get_recommendation': 'AI सिफारिश प्राप्त करें',
        'sandy': 'रेतीली',
        'loamy': 'दोमट',
        'clayey': 'चिकनी',
        'black': 'काली',
        'red': 'लाल',
        'rice': 'चावल',
        'wheat': 'गेहूं',
        'maize': 'मक्का',
        'cotton': 'कपास',
        'sugarcane': 'गन्ना',
        'pulses': 'दालें',
        'ground_nuts': 'मूंगफली',
        'millets': 'बाजरा',
        'no_fertilizer_needed': 'अतिरिक्त उर्वरक की आवश्यकता नहीं',
        'dose_per_acre': 'किग्रा प्रति एकड़',
        'confidence': 'विश्वास स्तर',
        'deficiency_analysis': 'पोषक तत्वों की कमी का विश्लेषण',
        'nitrogen_deficit': 'नाइट्रोजन',
        'phosphorous_deficit': 'फास्फोरस',
        'potassium_deficit': 'पोटेशियम',
        'requirement_fulfilled': 'आवश्यकता पूरी हुई',
        'download_report': 'रिपोर्ट डाउनलोड करें',
        'speak_result': 'सुनें',
        'fetching_location': 'स्थान प्राप्त कर रहा हूं...',
        'fetching_weather': 'मौसम डेटा प्राप्त कर रहा हूं...',
        'location_error': 'स्थान प्राप्त करने में असमर्थ',
        'weather_error': 'मौसम डेटा प्राप्त करने में विफल',
        'weather_fetched': 'मौसम डेटा सफलतापूर्वक प्राप्त हुआ!',
        'use_my_location': 'मेरा स्थान उपयोग करें',
        'analyzing': 'आपकी मिट्टी और फसल डेटा का विश्लेषण कर रहा हूं...',
        'feels_like': 'महसूस होता है',
        'kmh': 'किमी/घंटा',
        'air_quality_good': 'अच्छा',
        'air_quality_moderate': 'मध्यम',
        'air_quality_poor': 'खराब',
        'humidity_label': 'आर्द्रता',
        'wind_label': 'हवा',
        'uv_label': 'UV सूचकांक',
        'air_quality_label': 'वायु गुणवत्ता',
        'rain_label': 'बारिश',
        'sun_label': 'सूर्योदय/सूर्यास्त',
        'hourly_forecast': 'घंटेवार पूर्वानुमान',
        'weekly_forecast': '7-दिवसीय पूर्वानुमान',
        'history_title': 'भविष्यवाणी इतिहास',
        'all_crops': 'सभी फसलें',
        'download_pdf': 'PDF डाउनलोड करें',
        'date': 'तारीख',
        'fertilizer': 'उर्वरक',
        'dose': 'मात्रा (किग्रा)',
        'most_recommended': 'सबसे अनुशंसित',
        'avg_nitrogen': 'औसत नाइट्रोजन',
        'avg_phosphorous': 'औसत फास्फोरस',
        'avg_potassium': 'औसत पोटेशियम',
        'avg_confidence': 'औसत विश्वास',
        'avg_dose': 'औसत मात्रा',
        'most_grown_crop': 'सबसे उगाई जाने वाली फसल',
        'total_predictions': 'कुल भविष्यवाणियां',
        'predictions': 'भविष्यवाणियां',
        'fertilizer_distribution': 'उर्वरक वितरण',
        'crop_distribution': 'फसल वितरण',
        'monthly_trends': 'मासिक भविष्यवाणी रुझान',
        'confidence_trend': 'विश्वास रुझान',
        'nutrient_comparison': 'पोषक तत्वों की तुलना',
        'total_predictions_label': 'कुल भविष्यवाणियां',
        'most_recommended_label': 'सबसे अनुशंसित',
        'most_grown_label': 'सबसे उगाई जाने वाली फसल',
        'avg_nitrogen_label': 'औसत नाइट्रोजन',
        'avg_confidence_label': 'औसत विश्वास',
        'avg_dose_label': 'औसत मात्रा',
        'welcome_title': 'स्मार्ट कृषि AI प्लेटफॉर्म में आपका स्वागत है',
        'welcome_text': 'AI और वास्तविक समय मौसम डेटा द्वारा संचालित व्यक्तिगत उर्वरक सिफारिशें प्राप्त करें',
        'get_started': 'शुरू करें',
        'ai_greeting': 'नमस्ते! मैं आपका AI कृषि सहायक हूं। मैं फॉर्म भरने में आपकी मदद करूंगा। आपके पास जवाब देने के लिए 30 सेकंड हैं।',
        'ask_weather': 'क्या आप स्वचालित मौसम चाहते हैं? हां या ना कहें।',
        'ask_temperature': 'तापमान सेल्सियस में कितना है?',
        'ask_humidity': 'आर्द्रता प्रतिशत कितना है?',
        'ask_moisture': 'मिट्टी की नमी प्रतिशत कितनी है?',
        'ask_soil': 'आपकी मिट्टी का प्रकार क्या है? रेतीली, दोमट, चिकनी, काली, या लाल?',
        'ask_crop': 'आप कौन सी फसल उगा रहे हैं? चावल, गेहूं, मक्का, कपास, गन्ना, दालें, मूंगफली, या बाजरा?',
        'ask_nitrogen': 'आपकी मिट्टी में नाइट्रोजन का स्तर किग्रा/हेक्टेयर में कितना है?',
        'ask_phosphorous': 'आपकी मिट्टी में फास्फोरस का स्तर किग्रा/हेक्टेयर में कितना है?',
        'ask_potassium': 'आपकी मिट्टी में पोटेशियम का स्तर किग्रा/हेक्टेयर में कितना है?',
        'ai_thanks': 'धन्यवाद! आपकी जानकारी प्रोसेस कर रहा हूं...',
        'ai_no_response': 'कोई प्रतिक्रिया नहीं मिली। कृपया फिर से प्रयास करें।',
        'ai_stopping': 'कोई प्रतिक्रिया नहीं। AI सहायक बंद कर रहा हूं।',
        'ai_complete': 'सभी प्रश्न पूरे हुए! आपकी सिफारिश प्राप्त कर रहा हूं...',
        'ai_error': 'क्षमा करें, मैं समझ नहीं पाया। कृपया फिर से प्रयास करें।',
        'listening': 'सुन रहा हूं...',
        'speaking': 'बोल रहा हूं...',
        'processing': 'प्रोसेस कर रहा हूं...',
        'repeat': 'दोहराएं',
        'stop': 'रोकें',
        'yes_detected': 'हां का पता चला',
        'no_detected': 'ना का पता चला',
        'please_speak': 'कृपया अपना उत्तर बोलें',
        'time_remaining': 'शेष समय',
        'seconds': 'सेकंड',
        'recommended_fertilizer': 'अनुशंसित उर्वरक',
        'report_title': 'उर्वरक सिफारिश रिपोर्ट',
        'report_subtitle': 'AI-संचालित सटीक कृषि',
        'generated_on': 'जनरेट किया गया',
        'location': 'स्थान',
        'weather_conditions': 'मौसम की स्थिति',
        'soil_analysis': 'मिट्टी विश्लेषण',
        'current': 'वर्तमान',
        'required': 'आवश्यक',
        'ai_recommendation': 'AI सिफारिश',
        'explanation': 'व्याख्या',
        'application_instructions': 'आवेदन निर्देश',
        'safety_precautions': 'सुरक्षा सावधानियां',
        'irrigation_advice': 'सिंचाई सलाह',
        'application_timing': 'आवेदन समय',
        'application_method': 'आवेदन विधि',
        'storage_instructions': 'भंडारण निर्देश',
        'footer_text': 'स्मार्ट कृषि AI प्लेटफॉर्म - सतत भविष्य के लिए सटीक कृषि',
        'error_temperature_required': 'तापमान आवश्यक है',
        'error_temperature_range': f'तापमान {Config.TEMP_MIN}°C से {Config.TEMP_MAX}°C के बीच होना चाहिए',
        'error_humidity_required': 'आर्द्रता आवश्यक है',
        'error_humidity_range': f'आर्द्रता {Config.HUMIDITY_MIN}% से {Config.HUMIDITY_MAX}% के बीच होनी चाहिए',
        'error_moisture_required': 'मिट्टी की नमी आवश्यक है',
        'error_moisture_range': f'मिट्टी की नमी {Config.MOISTURE_MIN}% से {Config.MOISTURE_MAX}% के बीच होनी चाहिए',
        'error_soil_required': 'मिट्टी का प्रकार आवश्यक है',
        'error_crop_required': 'फसल का प्रकार आवश्यक है',
        'error_nitrogen_required': 'नाइट्रोजन स्तर आवश्यक है',
        'error_nitrogen_range': f'नाइट्रोजन {Config.NUTRIENT_MIN} से {Config.NUTRIENT_MAX} किग्रा/हेक्टेयर के बीच होना चाहिए',
        'error_phosphorous_required': 'फास्फोरस स्तर आवश्यक है',
        'error_phosphorous_range': f'फास्फोरस {Config.NUTRIENT_MIN} से {Config.NUTRIENT_MAX} किग्रा/हेक्टेयर के बीच होना चाहिए',
        'error_potassium_required': 'पोटेशियम स्तर आवश्यक है',
        'error_potassium_range': f'पोटेशियम {Config.NUTRIENT_MIN} से {Config.NUTRIENT_MAX} किग्रा/हेक्टेयर के बीच होना चाहिए',
        'error_invalid_soil': f'अमान्य मिट्टी का प्रकार। चुनें: {", ".join(Config.SOIL_TYPES)}',
        'error_invalid_crop': f'अमान्य फसल का प्रकार। चुनें: {", ".join(Config.CROP_TYPES)}',
        'error_missing_fields': 'आवश्यक फ़ील्ड गायब हैं',
        'error_prediction_service': 'भविष्यवाणी सेवा त्रुटि',
        'error_weather_service': 'मौसम सेवा त्रुटि',
        'error_network': 'नेटवर्क त्रुटि। कृपया पुनः प्रयास करें।',
        'optimization_title': 'अनुकूलित उर्वरक कार्यक्रम',
        'stage': 'चरण',
        'purpose': 'उद्देश्य',
        'time': 'समय',
        'total_nutrient_supply': 'कुल पोषक तत्व आपूर्ति',
        'optimization_score': 'अनुकूलन स्कोर',
        'soil_nutrient_balance': 'मिट्टी पोषक संतुलन',
        'optimal': 'इष्टतम',
        'irrigation_recommendation': 'सिंचाई सिफारिश',
        'summary': 'सारांश',
        'stage_1': 'आधार अनुप्रयोग',
        'stage_2': 'वानस्पतिक अवस्था',
        'stage_3': 'फूल आने की अवस्था',
        'purpose_1': 'जड़ विकास और प्रारंभिक वृद्धि',
        'purpose_2': 'वानस्पतिक वृद्धि और कल्ले निकलना',
        'purpose_3': 'फूल आना और फल विकास'
    },
    'te': {
        # Telugu translations (keep as in original)
        'app_name': 'స్మార్ట్ అగ్రికల్చర్ AI ప్లాట్‌ఫార్మ్',
        'tagline': 'AI-ఆధారిత ప్రెసిషన్ ఫార్మింగ్ అసిస్టెంట్',
        'home': 'హోమ్',
        'new_recommendation': 'కొత్త సిఫార్సు',
        'weather': 'వాతావరణం',
        'history': 'చరిత్ర',
        'analytics': 'అనలిటిక్స్',
        'input_parameters': 'ఇన్‌పుట్ పారామితులు',
        'auto_fetch': 'స్వయంచాలక వాతావరణం',
        'manual_entry': 'మాన్యువల్ ఎంట్రీ',
        'city_input': 'నగరం పేరు నమోదు చేయండి',
        'temperature': 'ఉష్ణోగ్రత (°C)',
        'humidity': 'తేమ (%)',
        'moisture': 'నేల తేమ (%)',
        'soil_type': 'నేల రకం',
        'crop_type': 'పంట రకం',
        'nitrogen': 'నైట్రోజన్ (N) - కిలో/హెక్టార్',
        'phosphorous': 'ఫాస్ఫరస్ (P) - కిలో/హెక్టార్',
        'potassium': 'పొటాషియం (K) - కిలో/హెక్టార్',
        'get_recommendation': 'AI సిఫార్సు పొందండి',
        'sandy': 'ఇసుక',
        'loamy': 'లోమీ',
        'clayey': 'బంకమట్టి',
        'black': 'నలుపు',
        'red': 'ఎరుపు',
        'rice': 'వరి',
        'wheat': 'గోధుమ',
        'maize': 'మొక్కజొన్న',
        'cotton': 'పత్తి',
        'sugarcane': 'చెరకు',
        'pulses': 'పప్పులు',
        'ground_nuts': 'వేరుశనగ',
        'millets': 'మిల్లెట్స్',
        'no_fertilizer_needed': 'అదనపు ఎరువులు అవసరం లేదు',
        'dose_per_acre': 'కిలో/ఎకరం',
        'confidence': 'విశ్వాసం',
        'deficiency_analysis': 'పోషకాల లోపం విశ్లేషణ',
        'nitrogen_deficit': 'నైట్రోజన్',
        'phosphorous_deficit': 'ఫాస్ఫరస్',
        'potassium_deficit': 'పొటాషియం',
        'requirement_fulfilled': 'అవసరం నెరవేరింది',
        'download_report': 'నివేదిక డౌన్‌లోడ్',
        'speak_result': 'వినండి',
        'fetching_location': 'స్థానం పొందుతోంది...',
        'fetching_weather': 'వాతావరణ డేటా పొందుతోంది...',
        'location_error': 'స్థానం పొందడం సాధ్యం కాలేదు',
        'weather_error': 'వాతావరణ డేటా పొందడం సాధ్యం కాలేదు',
        'weather_fetched': 'వాతావరణ డేటా విజయవంతంగా పొందబడింది!',
        'use_my_location': 'నా స్థానం ఉపయోగించు',
        'analyzing': 'మీ నేల మరియు పంట డేటాను విశ్లేషిస్తోంది...',
        'feels_like': 'అనుభూతి',
        'kmh': 'కిమీ/గం',
        'air_quality_good': 'మంచిది',
        'air_quality_moderate': 'మధ్యస్థం',
        'air_quality_poor': 'పేలవం',
        'humidity_label': 'తేమ',
        'wind_label': 'గాలి',
        'uv_label': 'UV సూచిక',
        'air_quality_label': 'గాలి నాణ్యత',
        'rain_label': 'వర్షం',
        'sun_label': 'సూర్యోదయం/సూర్యాస్తమయం',
        'hourly_forecast': 'గంటల వారీ అంచనా',
        'weekly_forecast': '7-రోజుల అంచనా',
        'history_title': 'అంచనాల చరిత్ర',
        'all_crops': 'అన్ని పంటలు',
        'download_pdf': 'PDF డౌన్‌లోడ్',
        'date': 'తేదీ',
        'fertilizer': 'ఎరువు',
        'dose': 'మోతాదు (కిలో)',
        'most_recommended': 'అత్యంత సిఫార్సు',
        'avg_nitrogen': 'సగటు నైట్రోజన్',
        'avg_phosphorous': 'సగటు ఫాస్ఫరస్',
        'avg_potassium': 'సగటు పొటాషియం',
        'avg_confidence': 'సగటు విశ్వాసం',
        'avg_dose': 'సగటు మోతాదు',
        'most_grown_crop': 'అత్యంత పండించే పంట',
        'total_predictions': 'మొత్తం అంచనాలు',
        'predictions': 'అంచనాలు',
        'fertilizer_distribution': 'ఎరువుల పంపిణీ',
        'crop_distribution': 'పంటల పంపిణీ',
        'monthly_trends': 'నెలవారీ అంచనాల ధోరణి',
        'confidence_trend': 'విశ్వాస ధోరణి',
        'nutrient_comparison': 'పోషకాల సగటుల పోలిక',
        'total_predictions_label': 'మొత్తం అంచనాలు',
        'most_recommended_label': 'అత్యంత సిఫార్సు',
        'most_grown_label': 'అత్యంత పండించే పంట',
        'avg_nitrogen_label': 'సగటు నైట్రోజన్',
        'avg_confidence_label': 'సగటు విశ్వాసం',
        'avg_dose_label': 'సగటు మోతాదు',
        'welcome_title': 'స్మార్ట్ అగ్రికల్చర్ AI ప్లాట్‌ఫార్మ్‌కు స్వాగతం',
        'welcome_text': 'AI మరియు నిజ-సమయ వాతావరణ డేటాతో వ్యక్తిగతీకరించిన ఎరువుల సిఫార్సులను పొందండి',
        'get_started': 'ప్రారంభించండి',
        'ai_greeting': 'నమస్కారం! నేను మీ AI వ్యవసాయ సహాయకుడిని. నేను మీకు ఫారమ్ నింపడంలో సహాయం చేస్తాను. మీకు సమాధానం ఇవ్వడానికి 30 సెకన్లు ఉన్నాయి.',
        'ask_weather': 'మీరు స్వయంచాలక వాతావరణం కావాలా? అవును లేదా కాదు అని చెప్పండి.',
        'ask_temperature': 'ఉష్ణోగ్రత సెల్సియస్‌లో ఎంత?',
        'ask_humidity': 'తేమ శాతం ఎంత?',
        'ask_moisture': 'నేల తేమ శాతం ఎంత?',
        'ask_soil': 'మీ నేల రకం ఏమిటి? ఇసుక, లోమీ, బంకమట్టి, నలుపు, లేదా ఎరుపు?',
        'ask_crop': 'మీరు ఏ పంట పండిస్తున్నారు? వరి, గోధుమ, మొక్కజొన్న, పత్తి, చెరకు, పప్పులు, వేరుశనగ, లేదా మిల్లెట్స్?',
        'ask_nitrogen': 'మీ నేలలో నైట్రోజన్ స్థాయి కిలో/హెక్టార్‌లో ఎంత?',
        'ask_phosphorous': 'మీ నేలలో ఫాస్ఫరస్ స్థాయి కిలో/హెక్టార్‌లో ఎంత?',
        'ask_potassium': 'మీ నేలలో పొటాషియం స్థాయి కిలో/హెక్టార్‌లో ఎంత?',
        'ai_thanks': 'ధన్యవాదాలు! మీ సమాచారాన్ని ప్రాసెస్ చేస్తున్నాను...',
        'ai_no_response': 'సమాధానం లేదు. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'ai_stopping': 'సమాధానం లేదు. AI సహాయకుడిని ఆపేస్తున్నాను.',
        'ai_complete': 'అన్ని ప్రశ్నలు పూర్తయ్యాయి! మీ సిఫార్సు పొందుతున్నాను...',
        'ai_error': 'క్షమించండి, నాకు అర్థం కాలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'listening': 'వింటున్నాను...',
        'speaking': 'మాట్లాడుతున్నాను...',
        'processing': 'ప్రాసెస్ చేస్తున్నాను...',
        'repeat': 'మళ్లీ చెప్పండి',
        'stop': 'ఆపు',
        'yes_detected': 'అవును గుర్తించబడింది',
        'no_detected': 'కాదు గుర్తించబడింది',
        'please_speak': 'దయచేసి మీ సమాధానం చెప్పండి',
        'time_remaining': 'మిగిలిన సమయం',
        'seconds': 'సెకన్లు',
        'recommended_fertilizer': 'సిఫార్సు చేసిన ఎరువు',
        'report_title': 'ఎరువుల సిఫార్సు నివేదిక',
        'report_subtitle': 'AI-ఆధారిత ప్రెసిషన్ ఫార్మింగ్',
        'generated_on': 'జనరేట్ చేయబడింది',
        'location': 'స్థానం',
        'weather_conditions': 'వాతావరణ పరిస్థితులు',
        'soil_analysis': 'నేల విశ్లేషణ',
        'current': 'ప్రస్తుత',
        'required': 'అవసరమైన',
        'ai_recommendation': 'AI సిఫార్సు',
        'explanation': 'వివరణ',
        'application_instructions': 'అప్లికేషన్ సూచనలు',
        'safety_precautions': 'భద్రతా జాగ్రత్తలు',
        'irrigation_advice': 'నీటిపారుదల సలహా',
        'application_timing': 'అప్లికేషన్ సమయం',
        'application_method': 'అప్లికేషన్ పద్ధతి',
        'storage_instructions': 'నిల్వ సూచనలు',
        'footer_text': 'స్మార్ట్ అగ్రికల్చర్ AI ప్లాట్ఫార్మ్ - సస్టైనబుల్ భవిష్యత్తు కోసం ప్రెసిషన్ ఫార్మింగ్',
        'error_temperature_required': 'ఉష్ణోగ్రత అవసరం',
        'error_temperature_range': f'ఉష్ణోగ్రత {Config.TEMP_MIN}°C నుండి {Config.TEMP_MAX}°C మధ్య ఉండాలి',
        'error_humidity_required': 'తేమ అవసరం',
        'error_humidity_range': f'తేమ {Config.HUMIDITY_MIN}% నుండి {Config.HUMIDITY_MAX}% మధ్య ఉండాలి',
        'error_moisture_required': 'నేల తేమ అవసరం',
        'error_moisture_range': f'నేల తేమ {Config.MOISTURE_MIN}% నుండి {Config.MOISTURE_MAX}% మధ్య ఉండాలి',
        'error_soil_required': 'నేల రకం అవసరం',
        'error_crop_required': 'పంట రకం అవసరం',
        'error_nitrogen_required': 'నైట్రోజన్ స్థాయి అవసరం',
        'error_nitrogen_range': f'నైట్రోజన్ {Config.NUTRIENT_MIN} నుండి {Config.NUTRIENT_MAX} కిలో/హెక్టార్ మధ్య ఉండాలి',
        'error_phosphorous_required': 'ఫాస్ఫరస్ స్థాయి అవసరం',
        'error_phosphorous_range': f'ఫాస్ఫరస్ {Config.NUTRIENT_MIN} నుండి {Config.NUTRIENT_MAX} కిలో/హెక్టార్ మధ్య ఉండాలి',
        'error_potassium_required': 'పొటాషియం స్థాయి అవసరం',
        'error_potassium_range': f'పొటాషియం {Config.NUTRIENT_MIN} నుండి {Config.NUTRIENT_MAX} కిలో/హెక్టార్ మధ్య ఉండాలి',
        'error_invalid_soil': f'చెల్లని నేల రకం. ఎంచుకోండి: {", ".join(Config.SOIL_TYPES)}',
        'error_invalid_crop': f'చెల్లని పంట రకం. ఎంచుకోండి: {", ".join(Config.CROP_TYPES)}',
        'error_missing_fields': 'తప్పనిసరి ఫీల్డ్లు లేవు',
        'error_prediction_service': 'ప్రిడిక్షన్ సేవ లోపం',
        'error_weather_service': 'వాతావరణ సేవ లోపం',
        'error_network': 'నెట్వర్క్ లోపం. దయచేసి మళ్లీ ప్రయత్నించండి.',
        'optimization_title': 'ఆప్టిమైజ్ చేసిన ఎరువుల షెడ్యూల్',
        'stage': 'దశ',
        'purpose': 'లక్ష్యం',
        'time': 'సమయం',
        'total_nutrient_supply': 'మొత్తం పోషక సరఫరా',
        'optimization_score': 'ఆప్టిమైజేషన్ స్కోర్',
        'soil_nutrient_balance': 'నేల పోషక సమతుల్యత',
        'optimal': 'సరైన',
        'irrigation_recommendation': 'నీటిపారుదల సిఫార్సు',
        'summary': 'సారాంశం',
        'stage_1': 'బేసల్ అప్లికేషన్',
        'stage_2': 'వెజిటేటివ్ దశ',
        'stage_3': 'పుష్పించే దశ',
        'purpose_1': 'రూట్ అభివృద్ధి మరియు ప్రారంభ పెరుగుదల',
        'purpose_2': 'వెజిటేటివ్ పెరుగుదల మరియు టిల్లరింగ్',
        'purpose_3': 'పుష్పించే మరియు ఫలాల అభివృద్ధి'
    },
    'ta': {
        # Tamil translations (keep as in original)
        'app_name': 'ஸ்மார்ட் அக்ரிகல்ச்சர் AI தளம்',
        'tagline': 'AI-இயக்கப்படும் துல்லிய விவசாய உதவியாளர்',
        'home': 'முகப்பு',
        'new_recommendation': 'புதிய பரிந்துரை',
        'weather': 'வானிலை',
        'history': 'வரலாறு',
        'analytics': 'பகுப்பாய்வு',
        'input_parameters': 'உள்ளீட்டு அளவுருக்கள்',
        'auto_fetch': 'தானியங்கி வானிலை',
        'manual_entry': 'கைமுறை உள்ளீடு',
        'city_input': 'நகரத்தின் பெயரை உள்ளிடவும்',
        'temperature': 'வெப்பநிலை (°C)',
        'humidity': 'ஈரப்பதம் (%)',
        'moisture': 'மண்ணின் ஈரப்பதம் (%)',
        'soil_type': 'மண் வகை',
        'crop_type': 'பயிர் வகை',
        'nitrogen': 'நைட்ரஜன் (N) - கிலோ/எக்டர்',
        'phosphorous': 'பாஸ்பரஸ் (P) - கிலோ/எக்டர்',
        'potassium': 'பொட்டாசியம் (K) - கிலோ/எக்டர்',
        'get_recommendation': 'AI பரிந்துரையைப் பெறுக',
        'sandy': 'மணல்மிகு',
        'loamy': 'களிமண்',
        'clayey': 'களிமண்',
        'black': 'கருப்பு',
        'red': 'சிவப்பு',
        'rice': 'நெல்',
        'wheat': 'கோதுமை',
        'maize': 'மக்காச்சோளம்',
        'cotton': 'பருத்தி',
        'sugarcane': 'கரும்பு',
        'pulses': 'பயறு',
        'ground_nuts': 'நிலக்கடலை',
        'millets': 'சிறுதானியங்கள்',
        'no_fertilizer_needed': 'கூடுதல் உரம் தேவையில்லை',
        'dose_per_acre': 'கிலோ/ஏக்கர்',
        'confidence': 'நம்பிக்கை',
        'deficiency_analysis': 'ஊட்டச்சத்து குறைபாடு பகுப்பாய்வு',
        'nitrogen_deficit': 'நைட்ரஜன்',
        'phosphorous_deficit': 'பாஸ்பரஸ்',
        'potassium_deficit': 'பொட்டாசியம்',
        'requirement_fulfilled': 'தேவை நிறைவேற்றப்பட்டது',
        'download_report': 'அறிக்கையைப் பதிவிறக்குக',
        'speak_result': 'கேளுங்கள்',
        'fetching_location': 'இருப்பிடத்தைப் பெறுகிறது...',
        'fetching_weather': 'வானிலைத் தரவைப் பெறுகிறது...',
        'location_error': 'இருப்பிடத்தைப் பெற முடியவில்லை',
        'weather_error': 'வானிலைத் தரவைப் பெற முடியவில்லை',
        'weather_fetched': 'வானிலைத் தரவு வெற்றிகரமாகப் பெறப்பட்டது!',
        'use_my_location': 'என் இருப்பிடத்தைப் பயன்படுத்து',
        'analyzing': 'உங்கள் மண் மற்றும் பயிர் தரவை பகுப்பாய்வு செய்கிறது...',
        'feels_like': 'உணரப்படுகிறது',
        'kmh': 'கிமீ/மணி',
        'air_quality_good': 'நல்லது',
        'air_quality_moderate': 'மிதமானது',
        'air_quality_poor': 'மோசமானது',
        'humidity_label': 'ஈரப்பதம்',
        'wind_label': 'காற்று',
        'uv_label': 'UV குறியீடு',
        'air_quality_label': 'காற்றின் தரம்',
        'rain_label': 'மழை',
        'sun_label': 'சூரிய உதயம்/சூரிய அஸ்தமனம்',
        'hourly_forecast': 'மணிநேர முன்னறிவிப்பு',
        'weekly_forecast': '7-நாள் முன்னறிவிப்பு',
        'history_title': 'முன்கணிப்பு வரலாறு',
        'all_crops': 'அனைத்து பயிர்கள்',
        'download_pdf': 'PDF பதிவிறக்குக',
        'date': 'தேதி',
        'fertilizer': 'உரம்',
        'dose': 'அளவு (கிலோ)',
        'most_recommended': 'மிகவும் பரிந்துரைக்கப்பட்டது',
        'avg_nitrogen': 'சராசரி நைட்ரஜன்',
        'avg_phosphorous': 'சராசரி பாஸ்பரஸ்',
        'avg_potassium': 'சராசரி பொட்டாசியம்',
        'avg_confidence': 'சராசரி நம்பிக்கை',
        'avg_dose': 'சராசரி அளவு',
        'most_grown_crop': 'அதிகம் விளைவிக்கப்படும் பயிர்',
        'total_predictions': 'மொத்த கணிப்புகள்',
        'predictions': 'கணிப்புகள்',
        'fertilizer_distribution': 'உர விநியோகம்',
        'crop_distribution': 'பயிர் விநியோகம்',
        'monthly_trends': 'மாதாந்திர கணிப்பு போக்குகள்',
        'confidence_trend': 'நம்பிக்கை போக்கு',
        'nutrient_comparison': 'ஊட்டச்சத்து சராசரிகள் ஒப்பீடு',
        'total_predictions_label': 'மொத்த கணிப்புகள்',
        'most_recommended_label': 'மிகவும் பரிந்துரைக்கப்பட்டது',
        'most_grown_label': 'அதிகம் விளைவிக்கப்படும் பயிர்',
        'avg_nitrogen_label': 'சராசரி நைட்ரஜன்',
        'avg_confidence_label': 'சராசரி நம்பிக்கை',
        'avg_dose_label': 'சராசரி அளவு',
        'welcome_title': 'ஸ்மார்ட் அக்ரிகல்ச்சர் AI தளத்திற்கு வரவேற்கிறோம்',
        'welcome_text': 'AI மற்றும் நிகழ்நேர வானிலை தரவுகளால் இயக்கப்படும் தனிப்பயனாக்கப்பட்ட உர பரிந்துரைகளைப் பெறுங்கள்',
        'get_started': 'தொடங்குங்கள்',
        'ai_greeting': 'வணக்கம்! நான் உங்கள் AI விவசாய உதவியாளர். படிவத்தை நிரப்ப உங்களுக்கு உதவுவேன். பதிலளிக்க உங்களுக்கு 30 வினாடிகள் உள்ளன.',
        'ask_weather': 'தானியங்கி வானிலையை விரும்புகிறீர்களா? ஆம் அல்லது இல்லை என்று கூறுங்கள்.',
        'ask_temperature': 'வெப்பநிலை செல்சியஸில் எவ்வளவு?',
        'ask_humidity': 'ஈரப்பதம் சதவீதம் எவ்வளவு?',
        'ask_moisture': 'மண்ணின் ஈரப்பதம் சதவீதம் எவ்வளவு?',
        'ask_soil': 'உங்கள் மண் வகை என்ன? மணல்மிகு, களிமண், களிமண், கருப்பு, அல்லது சிவப்பு?',
        'ask_crop': 'நீங்கள் என்ன பயிர் விளைவிக்கிறீர்கள்? நெல், கோதுமை, மக்காச்சோளம், பருத்தி, கரும்பு, பயறு, நிலக்கடலை, அல்லது சிறுதானியங்கள்?',
        'ask_nitrogen': 'உங்கள் மண்ணில் நைட்ரஜன் அளவு கிலோ/எக்டரில் எவ்வளவு?',
        'ask_phosphorous': 'உங்கள் மண்ணில் பாஸ்பரஸ் அளவு கிலோ/எக்டரில் எவ்வளவு?',
        'ask_potassium': 'உங்கள் மண்ணில் பொட்டாசியம் அளவு கிலோ/எக்டரில் எவ்வளவு?',
        'ai_thanks': 'நன்றி! உங்கள் தகவலை செயலாக்குகிறேன்...',
        'ai_no_response': 'பதில் இல்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.',
        'ai_stopping': 'பதில் இல்லை. AI உதவியாளரை நிறுத்துகிறேன்.',
        'ai_complete': 'அனைத்து கேள்விகளும் முடிந்தன! உங்கள் பரிந்துரையைப் பெறுகிறேன்...',
        'ai_error': 'மன்னிக்கவும், எனக்கு புரியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.',
        'listening': 'கேட்கிறேன்...',
        'speaking': 'பேசுகிறேன்...',
        'processing': 'செயலாக்குகிறேன்...',
        'repeat': 'மீண்டும் சொல்',
        'stop': 'நிறுத்து',
        'yes_detected': 'ஆம் கண்டறியப்பட்டது',
        'no_detected': 'இல்லை கண்டறியப்பட்டது',
        'please_speak': 'தயவுசெய்து உங்கள் பதிலைக் கூறுங்கள்',
        'time_remaining': 'மீதமுள்ள நேரம்',
        'seconds': 'வினாடிகள்',
        'recommended_fertilizer': 'பரிந்துரைக்கப்பட்ட உரம்',
        'report_title': 'உர பரிந்துரை அறிக்கை',
        'report_subtitle': 'AI-இயக்கப்படும் துல்லிய விவசாயம்',
        'generated_on': 'உருவாக்கப்பட்ட தேதி',
        'location': 'இடம்',
        'weather_conditions': 'வானிலை நிலைமைகள்',
        'soil_analysis': 'மண் பகுப்பாய்வு',
        'current': 'தற்போதைய',
        'required': 'தேவையான',
        'ai_recommendation': 'AI பரிந்துரை',
        'explanation': 'விளக்கம்',
        'application_instructions': 'பயன்பாட்டு வழிமுறைகள்',
        'safety_precautions': 'பாதுகாப்பு முன்னெச்சரிக்கைகள்',
        'irrigation_advice': 'நீர்ப்பாசன ஆலோசனை',
        'application_timing': 'பயன்பாட்டு நேரம்',
        'application_method': 'பயன்பாட்டு முறை',
        'storage_instructions': 'சேமிப்பு வழிமுறைகள்',
        'footer_text': 'ஸ்மார்ட் அக்ரிகல்ச்சர் AI தளம் - நிலையான எதிர்காலத்திற்கான துல்லிய விவசாயம்',
        'error_temperature_required': 'வெப்பநிலை தேவை',
        'error_temperature_range': f'வெப்பநிலை {Config.TEMP_MIN}°C மற்றும் {Config.TEMP_MAX}°C க்கு இடையில் இருக்க வேண்டும்',
        'error_humidity_required': 'ஈரப்பதம் தேவை',
        'error_humidity_range': f'ஈரப்பதம் {Config.HUMIDITY_MIN}% மற்றும் {Config.HUMIDITY_MAX}% க்கு இடையில் இருக்க வேண்டும்',
        'error_moisture_required': 'மண்ணின் ஈரப்பதம் தேவை',
        'error_moisture_range': f'மண்ணின் ஈரப்பதம் {Config.MOISTURE_MIN}% மற்றும் {Config.MOISTURE_MAX}% க்கு இடையில் இருக்க வேண்டும்',
        'error_soil_required': 'மண் வகை தேவை',
        'error_crop_required': 'பயிர் வகை தேவை',
        'error_nitrogen_required': 'நைட்ரஜன் அளவு தேவை',
        'error_nitrogen_range': f'நைட்ரஜன் {Config.NUTRIENT_MIN} மற்றும் {Config.NUTRIENT_MAX} கிலோ/எக்டர் இடையில் இருக்க வேண்டும்',
        'error_phosphorous_required': 'பாஸ்பரஸ் அளவு தேவை',
        'error_phosphorous_range': f'பாஸ்பரஸ் {Config.NUTRIENT_MIN} மற்றும் {Config.NUTRIENT_MAX} கிலோ/எக்டர் இடையில் இருக்க வேண்டும்',
        'error_potassium_required': 'பொட்டாசியம் அளவு தேவை',
        'error_potassium_range': f'பொட்டாசியம் {Config.NUTRIENT_MIN} மற்றும் {Config.NUTRIENT_MAX} கிலோ/எக்டர் இடையில் இருக்க வேண்டும்',
        'error_invalid_soil': f'தவறான மண் வகை. தேர்ந்தெடுக்கவும்: {", ".join(Config.SOIL_TYPES)}',
        'error_invalid_crop': f'தவறான பயிர் வகை. தேர்ந்தெடுக்கவும்: {", ".join(Config.CROP_TYPES)}',
        'error_missing_fields': 'தேவையான புலங்கள் இல்லை',
        'error_prediction_service': 'முன்கணிப்பு சேவை பிழை',
        'error_weather_service': 'வானிலை சேவை பிழை',
        'error_network': 'பிணைய பிழை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.',
        'optimization_title': 'உகந்த உர அட்டவணை',
        'stage': 'நிலை',
        'purpose': 'நோக்கம்',
        'time': 'நேரம்',
        'total_nutrient_supply': 'மொத்த ஊட்டச்சத்து வழங்கல்',
        'optimization_score': 'உகப்பாக்க மதிப்பெண்',
        'soil_nutrient_balance': 'மண் ஊட்டச்சத்து சமநிலை',
        'optimal': 'உகந்த',
        'irrigation_recommendation': 'நீர்ப்பாசன பரிந்துரை',
        'summary': 'சுருக்கம்',
        'stage_1': 'அடிப்படை பயன்பாடு',
        'stage_2': 'தாவர நிலை',
        'stage_3': 'பூக்கும் நிலை',
        'purpose_1': 'வேர் வளர்ச்சி மற்றும் ஆரம்ப வளர்ச்சி',
        'purpose_2': 'தாவர வளர்ச்சி மற்றும் கிளைத்தல்',
        'purpose_3': 'பூக்கும் மற்றும் பழ வளர்ச்சி'
    },
    'kn': {
        # Kannada translations (keep as in original)
        'app_name': 'ಸ್ಮಾರ್ಟ್ ಅಗ್ರಿಕಲ್ಚರ್ AI ಪ್ಲಾಟ್‌ಫಾರ್ಮ್',
        'tagline': 'AI-ಚಾಲಿತ ನಿಖರ ಕೃಷಿ ಸಹಾಯಕ',
        'home': 'ಮುಖಪುಟ',
        'new_recommendation': 'ಹೊಸ ಶಿಫಾರಸು',
        'weather': 'ಹವಾಮಾನ',
        'history': 'ಇತಿಹಾಸ',
        'analytics': 'ಅನಾಲಿಟಿಕ್ಸ್',
        'input_parameters': 'ಇನ್‌ಪುಟ್ ನಿಯತಾಂಕಗಳು',
        'auto_fetch': 'ಸ್ವಯಂಚಾಲಿತ ಹವಾಮಾನ',
        'manual_entry': 'ಹಸ್ತಚಾಲಿತ ನಮೂದು',
        'city_input': 'ನಗರದ ಹೆಸರನ್ನು ನಮೂದಿಸಿ',
        'temperature': 'ತಾಪಮಾನ (°C)',
        'humidity': 'ಆರ್ದ್ರತೆ (%)',
        'moisture': 'ಮಣ್ಣಿನ ತೇವಾಂಶ (%)',
        'soil_type': 'ಮಣ್ಣಿನ ಪ್ರಕಾರ',
        'crop_type': 'ಬೆಳೆ ಪ್ರಕಾರ',
        'nitrogen': 'ನೈಟ್ರೋಜನ್ (N) - ಕೆಜಿ/ಹೆಕ್ಟೇರ್',
        'phosphorous': 'ಫಾಸ್ಫರಸ್ (P) - ಕೆಜಿ/ಹೆಕ್ಟೇರ್',
        'potassium': 'ಪೊಟ್ಯಾಸಿಯಮ್ (K) - ಕೆಜಿ/ಹೆಕ್ಟೇರ್',
        'get_recommendation': 'AI ಶಿಫಾರಸು ಪಡೆಯಿರಿ',
        'sandy': 'ಮರಳು',
        'loamy': 'ಗೋಡು',
        'clayey': 'ಜೇಡಿ',
        'black': 'ಕಪ್ಪು',
        'red': 'ಕೆಂಪು',
        'rice': 'ಭತ್ತ',
        'wheat': 'ಗೋಧಿ',
        'maize': 'ಮೆಕ್ಕೆಜೋಳ',
        'cotton': 'ಹತ್ತಿ',
        'sugarcane': 'ಕಬ್ಬು',
        'pulses': 'ಬೇಳೆಕಾಳುಗಳು',
        'ground_nuts': 'ನೆಲಗಡಲೆ',
        'millets': 'ಸಿರಿಧಾನ್ಯಗಳು',
        'no_fertilizer_needed': 'ಹೆಚ್ಚುವರಿ ಗೊಬ್ಬರ ಅಗತ್ಯವಿಲ್ಲ',
        'dose_per_acre': 'ಕೆಜಿ/ಎಕರೆ',
        'confidence': 'ವಿಶ್ವಾಸ',
        'deficiency_analysis': 'ಪೋಷಕಾಂಶಗಳ ಕೊರತೆ ವಿಶ್ಲೇಷಣೆ',
        'nitrogen_deficit': 'ನೈಟ್ರೋಜನ್',
        'phosphorous_deficit': 'ಫಾಸ್ಫರಸ್',
        'potassium_deficit': 'ಪೊಟ್ಯಾಸಿಯಮ್',
        'requirement_fulfilled': 'ಅವಶ್ಯಕತೆ ಪೂರೈಸಲಾಗಿದೆ',
        'download_report': 'ವರದಿ ಡೌನ್‌ಲೋಡ್',
        'speak_result': 'ಕೇಳು',
        'fetching_location': 'ಸ್ಥಾನ ಪಡೆಯುತ್ತಿದೆ...',
        'fetching_weather': 'ಹವಾಮಾನ ಡೇಟಾ ಪಡೆಯುತ್ತಿದೆ...',
        'location_error': 'ಸ್ಥಾನ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ',
        'weather_error': 'ಹವಾಮಾನ ಡೇಟಾ ಪಡೆಯಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ',
        'weather_fetched': 'ಹವಾಮಾನ ಡೇಟಾ ಯಶಸ್ವಿಯಾಗಿ ಪಡೆಯಲಾಗಿದೆ!',
        'use_my_location': 'ನನ್ನ ಸ್ಥಾನ ಬಳಸು',
        'analyzing': 'ನಿಮ್ಮ ಮಣ್ಣು ಮತ್ತು ಬೆಳೆ ಡೇಟಾವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತಿದೆ...',
        'feels_like': 'ಅನುಭವ',
        'kmh': 'ಕಿಮೀ/ಗಂ',
        'air_quality_good': 'ಉತ್ತಮ',
        'air_quality_moderate': 'ಮಧ್ಯಮ',
        'air_quality_poor': 'ಕಳಪೆ',
        'humidity_label': 'ಆರ್ದ್ರತೆ',
        'wind_label': 'ಗಾಳಿ',
        'uv_label': 'UV ಸೂಚ್ಯಂಕ',
        'air_quality_label': 'ಗಾಳಿಯ ಗುಣಮಟ್ಟ',
        'rain_label': 'ಮಳೆ',
        'sun_label': 'ಸೂರ್ಯೋದಯ/ಸೂರ್ಯಾಸ್ತ',
        'hourly_forecast': 'ಗಂಟೆಯ ಮುನ್ಸೂಚನೆ',
        'weekly_forecast': '7-ದಿನಗಳ ಮುನ್ಸೂಚನೆ',
        'history_title': 'ಮುನ್ಸೂಚನೆ ಇತಿಹಾಸ',
        'all_crops': 'ಎಲ್ಲಾ ಬೆಳೆಗಳು',
        'download_pdf': 'PDF ಡೌನ್‌ಲೋಡ್',
        'date': 'ದಿನಾಂಕ',
        'fertilizer': 'ಗೊಬ್ಬರ',
        'dose': 'ಡೋಸ್ (ಕೆಜಿ)',
        'most_recommended': 'ಅತ್ಯಂತ ಶಿಫಾರಸು',
        'avg_nitrogen': 'ಸರಾಸರಿ ನೈಟ್ರೋಜನ್',
        'avg_phosphorous': 'ಸರಾಸರಿ ಫಾಸ್ಫರಸ್',
        'avg_potassium': 'ಸರಾಸರಿ ಪೊಟ್ಯಾಸಿಯಮ್',
        'avg_confidence': 'ಸರಾಸರಿ ವಿಶ್ವಾಸ',
        'avg_dose': 'ಸರಾಸರಿ ಡೋಸ್',
        'most_grown_crop': 'ಹೆಚ್ಚು ಬೆಳೆಯುವ ಬೆಳೆ',
        'total_predictions': 'ಒಟ್ಟು ಮುನ್ಸೂಚನೆಗಳು',
        'predictions': 'ಮುನ್ಸೂಚನೆಗಳು',
        'fertilizer_distribution': 'ಗೊಬ್ಬರ ವಿತರಣೆ',
        'crop_distribution': 'ಬೆಳೆ ವಿತರಣೆ',
        'monthly_trends': 'ಮಾಸಿಕ ಮುನ್ಸೂಚನೆ ಪ್ರವೃತ್ತಿಗಳು',
        'confidence_trend': 'ವಿಶ್ವಾಸ ಪ್ರವೃತ್ತಿ',
        'nutrient_comparison': 'ಪೋಷಕಾಂಶಗಳ ಸರಾಸರಿ ಹೋಲಿಕೆ',
        'total_predictions_label': 'ಒಟ್ಟು ಮುನ್ಸೂಚನೆಗಳು',
        'most_recommended_label': 'ಅತ್ಯಂತ ಶಿಫಾರಸು',
        'most_grown_label': 'ಹೆಚ್ಚು ಬೆಳೆಯುವ ಬೆಳೆ',
        'avg_nitrogen_label': 'ಸರಾಸರಿ ನೈಟ್ರೋಜನ್',
        'avg_confidence_label': 'ಸರಾಸರಿ ವಿಶ್ವಾಸ',
        'avg_dose_label': 'ಸರಾಸರಿ ಡೋಸ್',
        'welcome_title': 'ಸ್ಮಾರ್ಟ್ ಅಗ್ರಿಕಲ್ಚರ್ AI ಪ್ಲಾಟ್‌ಫಾರ್ಮ್‌ಗೆ ಸುಸ್ವಾಗತ',
        'welcome_text': 'AI ಮತ್ತು ನೈಜ-ಸಮಯದ ಹವಾಮಾನ ಡೇಟಾದಿಂದ ವೈಯಕ್ತೀಕರಿಸಿದ ಗೊಬ್ಬರ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ',
        'get_started': 'ಪ್ರಾರಂಭಿಸಿ',
        'ai_greeting': 'ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ AI ಕೃಷಿ ಸಹಾಯಕ. ಫಾರ್ಮ್ ತುಂಬಲು ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ಉತ್ತರಿಸಲು ನಿಮಗೆ 30 ಸೆಕೆಂಡುಗಳಿವೆ.',
        'ask_weather': 'ನೀವು ಸ್ವಯಂಚಾಲಿತ ಹವಾಮಾನ ಬಯಸುವಿರಾ? ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.',
        'ask_temperature': 'ತಾಪಮಾನ ಸೆಲ್ಸಿಯಸ್‌ನಲ್ಲಿ ಎಷ್ಟು?',
        'ask_humidity': 'ಆರ್ದ್ರತೆ ಶೇಕಡಾ ಎಷ್ಟು?',
        'ask_moisture': 'ಮಣ್ಣಿನ ತೇವಾಂಶ ಶೇಕಡಾ ಎಷ್ಟು?',
        'ask_soil': 'ನಿಮ್ಮ ಮಣ್ಣಿನ ಪ್ರಕಾರ ಯಾವುದು? ಮರಳು, ಗೋಡು, ಜೇಡಿ, ಕಪ್ಪು, ಅಥವಾ ಕೆಂಪು?',
        'ask_crop': 'ನೀವು ಯಾವ ಬೆಳೆಯನ್ನು ಬೆಳೆಯುತ್ತಿದ್ದೀರಿ? ಭತ್ತ, ಗೋಧಿ, ಮೆಕ್ಕೆಜೋಳ, ಹತ್ತಿ, ಕಬ್ಬು, ಬೇಳೆಕಾಳುಗಳು, ನೆಲಗಡಲೆ, ಅಥವಾ ಸಿರಿಧಾನ್ಯಗಳು?',
        'ask_nitrogen': 'ನಿಮ್ಮ ಮಣ್ಣಿನಲ್ಲಿ ನೈಟ್ರೋಜನ್ ಮಟ್ಟ ಕೆಜಿ/ಹೆಕ್ಟೇರ್‌ನಲ್ಲಿ ಎಷ್ಟು?',
        'ask_phosphorous': 'ನಿಮ್ಮ ಮಣ್ಣಿನಲ್ಲಿ ಫಾಸ್ಫರಸ್ ಮಟ್ಟ ಕೆಜಿ/ಹೆಕ್ಟೇರ್‌ನಲ್ಲಿ ಎಷ್ಟು?',
        'ask_potassium': 'ನಿಮ್ಮ ಮಣ್ಣಿನಲ್ಲಿ ಪೊಟ್ಯಾಸಿಯಮ್ ಮಟ್ಟ ಕೆಜಿ/ಹೆಕ್ಟೇರ್‌ನಲ್ಲಿ ಎಷ್ಟು?',
        'ai_thanks': 'ಧನ್ಯವಾದಗಳು! ನಿಮ್ಮ ಮಾಹಿತಿಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸುತ್ತಿದ್ದೇನೆ...',
        'ai_no_response': 'ಯಾವುದೇ ಪ್ರತಿಕ್ರಿಯೆ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
        'ai_stopping': 'ಯಾವುದೇ ಪ್ರತಿಕ್ರಿಯೆ ಇಲ್ಲ. AI ಸಹಾಯಕವನ್ನು ನಿಲ್ಲಿಸುತ್ತಿದ್ದೇನೆ.',
        'ai_complete': 'ಎಲ್ಲಾ ಪ್ರಶ್ನೆಗಳು ಪೂರ್ಣಗೊಂಡಿವೆ! ನಿಮ್ಮ ಶಿಫಾರಸು ಪಡೆಯುತ್ತಿದ್ದೇನೆ...',
        'ai_error': 'ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
        'listening': 'ಕೇಳುತ್ತಿದ್ದೇನೆ...',
        'speaking': 'ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ...',
        'processing': 'ಪ್ರಕ್ರಿಯೆಗೊಳಿಸುತ್ತಿದ್ದೇನೆ...',
        'repeat': 'ಪುನರಾವರ್ತಿಸಿ',
        'stop': 'ನಿಲ್ಲಿಸು',
        'yes_detected': 'ಹೌದು ಪತ್ತೆಯಾಗಿದೆ',
        'no_detected': 'ಇಲ್ಲ ಪತ್ತೆಯಾಗಿದೆ',
        'please_speak': 'ದಯವಿಟ್ಟು ನಿಮ್ಮ ಉತ್ತರವನ್ನು ಹೇಳಿ',
        'time_remaining': 'ಉಳಿದ ಸಮಯ',
        'seconds': 'ಸೆಕೆಂಡುಗಳು',
        'recommended_fertilizer': 'ಶಿಫಾರಸು ಮಾಡಿದ ಗೊಬ್ಬರ',
        'report_title': 'ಗೊಬ್ಬರ ಶಿಫಾರಸು ವರದಿ',
        'report_subtitle': 'AI-ಚಾಲಿತ ನಿಖರ ಕೃಷಿ',
        'generated_on': 'ರಚಿಸಲಾಗಿದೆ',
        'location': 'ಸ್ಥಳ',
        'weather_conditions': 'ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳು',
        'soil_analysis': 'ಮಣ್ಣಿನ ವಿಶ್ಲೇಷಣೆ',
        'current': 'ಪ್ರಸ್ತುತ',
        'required': 'ಅಗತ್ಯವಿದೆ',
        'ai_recommendation': 'AI ಶಿಫಾರಸು',
        'explanation': 'ವಿವರಣೆ',
        'application_instructions': 'ಅಪ್ಲಿಕೇಶನ್ ಸೂಚನೆಗಳು',
        'safety_precautions': 'ಸುರಕ್ಷತಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು',
        'irrigation_advice': 'ನೀರಾವರಿ ಸಲಹೆ',
        'application_timing': 'ಅಪ್ಲಿಕೇಶನ್ ಸಮಯ',
        'application_method': 'ಅಪ್ಲಿಕೇಶನ್ ವಿಧಾನ',
        'storage_instructions': 'ಶೇಖರಣಾ ಸೂಚನೆಗಳು',
        'footer_text': 'ಸ್ಮಾರ್ಟ್ ಅಗ್ರಿಕಲ್ಚರ್ AI ಪ್ಲಾಟ್ಫಾರ್ಮ್ - ಸುಸ್ಥಿರ ಭವಿಷ್ಯಕ್ಕಾಗಿ ನಿಖರ ಕೃಷಿ',
        'error_temperature_required': 'ತಾಪಮಾನ ಅಗತ್ಯವಿದೆ',
        'error_temperature_range': f'ತಾಪಮಾನ {Config.TEMP_MIN}°C ಮತ್ತು {Config.TEMP_MAX}°C ನಡುವೆ ಇರಬೇಕು',
        'error_humidity_required': 'ಆರ್ದ್ರತೆ ಅಗತ್ಯವಿದೆ',
        'error_humidity_range': f'ಆರ್ದ್ರತೆ {Config.HUMIDITY_MIN}% ಮತ್ತು {Config.HUMIDITY_MAX}% ನಡುವೆ ಇರಬೇಕು',
        'error_moisture_required': 'ಮಣ್ಣಿನ ತೇವಾಂಶ ಅಗತ್ಯವಿದೆ',
        'error_moisture_range': f'ಮಣ್ಣಿನ ತೇವಾಂಶ {Config.MOISTURE_MIN}% ಮತ್ತು {Config.MOISTURE_MAX}% ನಡುವೆ ಇರಬೇಕು',
        'error_soil_required': 'ಮಣ್ಣಿನ ಪ್ರಕಾರ ಅಗತ್ಯವಿದೆ',
        'error_crop_required': 'ಬೆಳೆ ಪ್ರಕಾರ ಅಗತ್ಯವಿದೆ',
        'error_nitrogen_required': 'ನೈಟ್ರೋಜನ್ ಮಟ್ಟ ಅಗತ್ಯವಿದೆ',
        'error_nitrogen_range': f'ನೈಟ್ರೋಜನ್ {Config.NUTRIENT_MIN} ಮತ್ತು {Config.NUTRIENT_MAX} ಕೆಜಿ/ಹೆಕ್ಟೇರ್ ನಡುವೆ ಇರಬೇಕು',
        'error_phosphorous_required': 'ಫಾಸ್ಫರಸ್ ಮಟ್ಟ ಅಗತ್ಯವಿದೆ',
        'error_phosphorous_range': f'ಫಾಸ್ಫರಸ್ {Config.NUTRIENT_MIN} ಮತ್ತು {Config.NUTRIENT_MAX} ಕೆಜಿ/ಹೆಕ್ಟೇರ್ ನಡುವೆ ಇರಬೇಕು',
        'error_potassium_required': 'ಪೊಟ್ಯಾಸಿಯಮ್ ಮಟ್ಟ ಅಗತ್ಯವಿದೆ',
        'error_potassium_range': f'ಪೊಟ್ಯಾಸಿಯಮ್ {Config.NUTRIENT_MIN} ಮತ್ತು {Config.NUTRIENT_MAX} ಕೆಜಿ/ಹೆಕ್ಟೇರ್ ನಡುವೆ ಇರಬೇಕು',
        'error_invalid_soil': f'ಅಮಾನ್ಯ ಮಣ್ಣಿನ ಪ್ರಕಾರ. ಆಯ್ಕೆಮಾಡಿ: {", ".join(Config.SOIL_TYPES)}',
        'error_invalid_crop': f'ಅಮಾನ್ಯ ಬೆಳೆ ಪ್ರಕಾರ. ಆಯ್ಕೆಮಾಡಿ: {", ".join(Config.CROP_TYPES)}',
        'error_missing_fields': 'ಅಗತ್ಯ ಕ್ಷೇತ್ರಗಳು ಕಾಣೆಯಾಗಿವೆ',
        'error_prediction_service': 'ಮುನ್ಸೂಚನೆ ಸೇವಾ ದೋಷ',
        'error_weather_service': 'ಹವಾಮಾನ ಸೇವಾ ದೋಷ',
        'error_network': 'ನೆಟ್‌ವರ್ಕ್ ದೋಷ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
        'optimization_title': 'ಆಪ್ಟಿಮೈಸ್ಡ್ ಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ',
        'stage': 'ಹಂತ',
        'purpose': 'ಉದ್ದೇಶ',
        'time': 'ಸಮಯ',
        'total_nutrient_supply': 'ಒಟ್ಟು ಪೋಷಕಾಂಶ ಪೂರೈಕೆ',
        'optimization_score': 'ಆಪ್ಟಿಮೈಸೇಶನ್ ಸ್ಕೋರ್',
        'soil_nutrient_balance': 'ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ ಸಮತೋಲನ',
        'optimal': 'ಸೂಕ್ತ',
        'irrigation_recommendation': 'ನೀರಾವರಿ ಶಿಫಾರಸು',
        'summary': 'ಸಾರಾಂಶ',
        'stage_1': 'ಮೂಲ ಅಪ್ಲಿಕೇಶನ್',
        'stage_2': 'ಸಸ್ಯಕ ಹಂತ',
        'stage_3': 'ಹೂಬಿಡುವ ಹಂತ',
        'purpose_1': 'ಬೇರು ಅಭಿವೃದ್ಧಿ ಮತ್ತು ಆರಂಭಿಕ ಬೆಳವಣಿಗೆ',
        'purpose_2': 'ಸಸ್ಯಕ ಬೆಳವಣಿಗೆ ಮತ್ತು ಕವಲೊಡೆಯುವಿಕೆ',
        'purpose_3': 'ಹೂಬಿಡುವಿಕೆ ಮತ್ತು ಫಲ ಅಭಿವೃದ್ಧಿ'
    }
}

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    """Database operations"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database with all required tables and migrate if needed"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Check if predictions table exists
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
                table_exists = c.fetchone() is not None
                
                if not table_exists:
                    # Create predictions table with optimization fields
                    c.execute('''CREATE TABLE predictions
                                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                 date TEXT, location TEXT, city TEXT, country TEXT,
                                 crop TEXT, soil TEXT, fertilizer TEXT, dose REAL,
                                 confidence REAL, nitrogen REAL, phosphorous REAL,
                                 potassium REAL, temperature REAL, humidity REAL,
                                 moisture REAL, weather_condition TEXT,
                                 language TEXT, 
                                 optimization_score INTEGER,
                                 soil_nutrient_balance TEXT,
                                 optimization_summary TEXT,
                                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                    logger.info("Created predictions table with optimization columns")
                else:
                    # Check existing columns and add missing ones
                    logger.info("Predictions table exists, checking columns...")
                    
                    # Get existing columns
                    c.execute("PRAGMA table_info(predictions)")
                    existing_columns = [column[1] for column in c.fetchall()]
                    logger.info(f"Existing columns: {existing_columns}")
                    
                    # Add missing columns if they don't exist
                    if 'optimization_score' not in existing_columns:
                        try:
                            c.execute("ALTER TABLE predictions ADD COLUMN optimization_score INTEGER")
                            logger.info("Added optimization_score column to predictions table")
                        except Exception as e:
                            logger.warning(f"Could not add optimization_score column: {e}")
                    
                    if 'soil_nutrient_balance' not in existing_columns:
                        try:
                            c.execute("ALTER TABLE predictions ADD COLUMN soil_nutrient_balance TEXT")
                            logger.info("Added soil_nutrient_balance column to predictions table")
                        except Exception as e:
                            logger.warning(f"Could not add soil_nutrient_balance column: {e}")
                    
                    if 'optimization_summary' not in existing_columns:
                        try:
                            c.execute("ALTER TABLE predictions ADD COLUMN optimization_summary TEXT")
                            logger.info("Added optimization_summary column to predictions table")
                        except Exception as e:
                            logger.warning(f"Could not add optimization_summary column: {e}")
                
                # Create weather cache table
                c.execute('''CREATE TABLE IF NOT EXISTS weather_cache
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             city TEXT, country TEXT, data TEXT,
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                
                # Create optimization cache table
                c.execute('''CREATE TABLE IF NOT EXISTS optimization_cache
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             crop TEXT, soil_type TEXT, nitrogen REAL,
                             phosphorous REAL, potassium REAL,
                             predicted_fertilizer TEXT, base_dose REAL,
                             optimization_data TEXT,
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                
                # Create indexes for faster queries
                c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                            ON predictions(timestamp)''')
                c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_crop 
                            ON predictions(crop)''')
                c.execute('''CREATE INDEX IF NOT EXISTS idx_predictions_date 
                            ON predictions(date)''')
                c.execute('''CREATE INDEX IF NOT EXISTS idx_weather_cache_city 
                            ON weather_cache(city)''')
                c.execute('''CREATE INDEX IF NOT EXISTS idx_optimization_cache 
                            ON optimization_cache(crop, soil_type, nitrogen, phosphorous, potassium)''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
                # Verify final structure
                c.execute("PRAGMA table_info(predictions)")
                final_columns = [column[1] for column in c.fetchall()]
                logger.info(f"Final columns in predictions table: {final_columns}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_prediction(self, data):
        """Save prediction to database with optimization data"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Extract optimization data if available
                optimization_score = data.get('optimization', {}).get('optimization_score') if data.get('optimization') else None
                soil_nutrient_balance = data.get('optimization', {}).get('soil_nutrient_balance') if data.get('optimization') else None
                optimization_summary = data.get('optimization', {}).get('summary') if data.get('optimization') else None
                
                c.execute('''INSERT INTO predictions 
                            (date, location, city, country, crop, soil, fertilizer, dose, confidence,
                             nitrogen, phosphorous, potassium, temperature, humidity, moisture,
                             weather_condition, language, optimization_score, soil_nutrient_balance, optimization_summary)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (data['date'], data['location'], data.get('city', ''), data.get('country', ''),
                          data['crop'], data['soil'], data['fertilizer'], data['dose'], data['confidence'],
                          data['nitrogen'], data['phosphorous'], data['potassium'],
                          data['temperature'], data['humidity'], data['moisture'],
                          data.get('weather_condition', ''), data.get('language', 'en'),
                          optimization_score, soil_nutrient_balance, optimization_summary))
                conn.commit()
                
                # Get the ID of the inserted record
                prediction_id = c.lastrowid
                logger.info(f"Prediction {prediction_id} saved to database for {data['crop']} at {data['location']}")
                return prediction_id
        except Exception as e:
            logger.error(f"Failed to save prediction to database: {e}")
            return None
    
    def save_optimization(self, crop, soil_type, nitrogen, phosphorous, potassium, 
                         predicted_fertilizer, base_dose, optimization_data):
        """Save optimization result to cache"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO optimization_cache 
                            (crop, soil_type, nitrogen, phosphorous, potassium,
                             predicted_fertilizer, base_dose, optimization_data)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (crop, soil_type, nitrogen, phosphorous, potassium,
                          predicted_fertilizer, base_dose, json.dumps(optimization_data)))
                conn.commit()
                logger.info(f"Optimization saved to cache for {crop}")
                return True
        except Exception as e:
            logger.error(f"Failed to save optimization to cache: {e}")
            return False
    
    def get_optimization_cache(self, crop, soil_type, nitrogen, phosphorous, potassium):
        """Get cached optimization result"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''SELECT optimization_data FROM optimization_cache 
                            WHERE crop=? AND soil_type=? AND nitrogen=? AND phosphorous=? AND potassium=?
                            AND timestamp > datetime('now', '-7 day')
                            ORDER BY timestamp DESC LIMIT 1''',
                         (crop, soil_type, nitrogen, phosphorous, potassium))
                cached = c.fetchone()
                if cached:
                    return json.loads(cached[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get optimization cache: {e}")
            return None
    
    def get_history(self, limit=None, crop=None, date=None, offset=0):
        """
        Get prediction history with pagination support
        
        Args:
            limit (int, optional): Number of records to return. None returns all records.
            crop (str, optional): Filter by crop type
            date (str, optional): Filter by date
            offset (int, optional): Offset for pagination
        
        Returns:
            dict: Dictionary with 'data', 'total', 'offset', 'limit' keys
        """
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Build base query
                query = "SELECT * FROM predictions WHERE 1=1"
                params = []
                
                # Add filters
                if crop and crop != 'All Crops' and crop != '':
                    query += " AND crop = ?"
                    params.append(crop)
                
                if date:
                    query += " AND date = ?"
                    params.append(date)
                
                # Get total count for pagination info
                count_query = query.replace("SELECT *", "SELECT COUNT(*)")
                c.execute(count_query, params)
                total_count = c.fetchone()[0]
                
                # Add ordering and pagination
                query += " ORDER BY timestamp DESC"
                
                if limit is not None:
                    query += " LIMIT ? OFFSET ?"
                    params.append(limit)
                    params.append(offset)
                
                c.execute(query, params)
                results = [dict(row) for row in c.fetchall()]
                
                return {
                    'data': results,
                    'total': total_count,
                    'offset': offset,
                    'limit': limit
                }
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return {
                'data': [],
                'total': 0,
                'offset': offset,
                'limit': limit
            }
    
    def get_history_stats(self):
        """Get prediction history statistics"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                c.execute("SELECT COUNT(*) FROM predictions")
                count = c.fetchone()[0]
                
                if count == 0:
                    return {
                        'recent': [], 
                        'most_common': 'N/A', 
                        'avg_dose': 0,
                        'total_count': 0
                    }
                
                # Get recent predictions with optimization data
                c.execute("""SELECT id, date, crop, soil, fertilizer, dose, confidence,
                                   optimization_score, soil_nutrient_balance
                            FROM predictions 
                            ORDER BY timestamp DESC 
                            LIMIT 10""")
                recent = [dict(row) for row in c.fetchall()]
                
                # Most recommended fertilizer
                c.execute("SELECT fertilizer, COUNT(*) as count FROM predictions GROUP BY fertilizer ORDER BY count DESC LIMIT 1")
                most_common_row = c.fetchone()
                most_common = most_common_row['fertilizer'] if most_common_row else 'N/A'
                
                # Average dose
                c.execute("SELECT AVG(dose) FROM predictions")
                avg_dose = c.fetchone()[0] or 0
                
                return {
                    'recent': recent,
                    'most_common': most_common,
                    'avg_dose': round(avg_dose, 2),
                    'total_count': count
                }
        except Exception as e:
            logger.error(f"Failed to get history stats: {e}")
            return {
                'recent': [], 
                'most_common': 'N/A', 
                'avg_dose': 0,
                'total_count': 0
            }
    
    def get_history_by_id(self, prediction_id):
        """Get a single prediction by ID"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
                row = c.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get prediction by id {prediction_id}: {e}")
            return None
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction by ID"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
                conn.commit()
                logger.info(f"Deleted prediction with id {prediction_id}")
                return c.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete prediction {prediction_id}: {e}")
            return False
    
    def get_analytics(self, days=30):
        """Get comprehensive analytics data with time range"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                
                # Get date range for filtering
                if days:
                    date_filter = f"date > date('now', '-{days} days')"
                else:
                    date_filter = "1=1"
                
                # Total predictions in range
                c.execute(f"SELECT COUNT(*) FROM predictions WHERE {date_filter}")
                total_predictions = c.fetchone()[0] or 0
                
                if total_predictions == 0:
                    return {
                        'total_predictions': 0,
                        'most_recommended': '-',
                        'most_crop': '-',
                        'avg_nitrogen': 0,
                        'avg_phosphorous': 0,
                        'avg_potassium': 0,
                        'avg_confidence': 0,
                        'avg_dose': 0,
                        'avg_optimization_score': 0,
                        'fertilizer_distribution': [],
                        'crop_distribution': [],
                        'monthly_trends': [],
                        'confidence_trend': [],
                        'optimization_trend': [],
                        'date_range': days
                    }
                
                # Most recommended fertilizer
                c.execute(f"""SELECT fertilizer, COUNT(*) as count 
                            FROM predictions 
                            WHERE {date_filter} 
                            GROUP BY fertilizer 
                            ORDER BY count DESC LIMIT 1""")
                most_recommended_row = c.fetchone()
                most_recommended = most_recommended_row['fertilizer'] if most_recommended_row else '-'
                
                # Most grown crop
                c.execute(f"""SELECT crop, COUNT(*) as count 
                            FROM predictions 
                            WHERE {date_filter} 
                            GROUP BY crop 
                            ORDER BY count DESC LIMIT 1""")
                most_crop_row = c.fetchone()
                most_crop = most_crop_row['crop'] if most_crop_row else '-'
                
                # Averages
                c.execute(f"""SELECT AVG(nitrogen), AVG(phosphorous), AVG(potassium), 
                                    AVG(confidence), AVG(dose), AVG(optimization_score)
                            FROM predictions 
                            WHERE {date_filter}""")
                avg_row = c.fetchone()
                avg_nitrogen = round(avg_row[0] or 0, 2)
                avg_phosphorous = round(avg_row[1] or 0, 2)
                avg_potassium = round(avg_row[2] or 0, 2)
                avg_confidence = round(avg_row[3] or 0, 2)
                avg_dose = round(avg_row[4] or 0, 2)
                avg_optimization_score = round(avg_row[5] or 0, 2)
                
                # Fertilizer distribution
                c.execute(f"""SELECT fertilizer, COUNT(*) as count 
                            FROM predictions 
                            WHERE {date_filter} 
                            GROUP BY fertilizer 
                            ORDER BY count DESC""")
                fertilizer_dist = [{"name": row['fertilizer'], "value": row['count']} for row in c.fetchall()]
                
                # Crop distribution
                c.execute(f"""SELECT crop, COUNT(*) as count 
                            FROM predictions 
                            WHERE {date_filter} 
                            GROUP BY crop 
                            ORDER BY count DESC""")
                crop_dist = [{"name": row['crop'], "value": row['count']} for row in c.fetchall()]
                
                # Monthly trends
                c.execute(f"""SELECT strftime('%Y-%m', date) as month, COUNT(*) as count 
                            FROM predictions 
                            WHERE {date_filter}
                            GROUP BY month 
                            ORDER BY month DESC 
                            LIMIT 12""")
                monthly_rows = c.fetchall()
                monthly_trends = [{"month": row['month'], "count": row['count']} for row in monthly_rows]
                monthly_trends.reverse()
                
                # Confidence trend by month
                c.execute(f"""SELECT strftime('%Y-%m', date) as month, AVG(confidence) as avg_conf
                            FROM predictions 
                            WHERE {date_filter}
                            GROUP BY month 
                            ORDER BY month DESC 
                            LIMIT 12""")
                conf_rows = c.fetchall()
                confidence_trend = [{"month": row['month'], "value": round(row['avg_conf'], 2)} for row in conf_rows]
                confidence_trend.reverse()
                
                # Optimization score trend by month
                c.execute(f"""SELECT strftime('%Y-%m', date) as month, AVG(optimization_score) as avg_score
                            FROM predictions 
                            WHERE {date_filter} AND optimization_score IS NOT NULL
                            GROUP BY month 
                            ORDER BY month DESC 
                            LIMIT 12""")
                opt_rows = c.fetchall()
                optimization_trend = [{"month": row['month'], "value": round(row['avg_score'], 2)} for row in opt_rows]
                optimization_trend.reverse()
                
                return {
                    'total_predictions': total_predictions,
                    'most_recommended': most_recommended,
                    'most_crop': most_crop,
                    'avg_nitrogen': avg_nitrogen,
                    'avg_phosphorous': avg_phosphorous,
                    'avg_potassium': avg_potassium,
                    'avg_confidence': avg_confidence,
                    'avg_dose': avg_dose,
                    'avg_optimization_score': avg_optimization_score,
                    'fertilizer_distribution': fertilizer_dist,
                    'crop_distribution': crop_dist,
                    'monthly_trends': monthly_trends,
                    'confidence_trend': confidence_trend,
                    'optimization_trend': optimization_trend,
                    'date_range': days
                }
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {
                'total_predictions': 0,
                'most_recommended': '-',
                'most_crop': '-',
                'avg_nitrogen': 0,
                'avg_phosphorous': 0,
                'avg_potassium': 0,
                'avg_confidence': 0,
                'avg_dose': 0,
                'avg_optimization_score': 0,
                'fertilizer_distribution': [],
                'crop_distribution': [],
                'monthly_trends': [],
                'confidence_trend': [],
                'optimization_trend': [],
                'date_range': days
            }
    
    def get_weather_cache(self, city):
        """Get cached weather data"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("""SELECT data FROM weather_cache 
                            WHERE city=? AND timestamp > datetime('now', '-1 hour') 
                            ORDER BY timestamp DESC LIMIT 1""", (city.lower(),))
                cached = c.fetchone()
                if cached:
                    return json.loads(cached[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get weather cache: {e}")
            return None
    
    def save_weather_cache(self, city, country, data):
        """Save weather data to cache"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("INSERT INTO weather_cache (city, country, data) VALUES (?, ?, ?)",
                         (city.lower(), country, json.dumps(data)))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save weather cache: {e}")
            return False

# Initialize database
db = Database(Config.DATABASE)
db.init_db()

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelService:
    """ML model service"""
    
    def __init__(self):
        self.model = None
        self.target_encoder = None
        self.load_models()
    
    def load_model_safe(self, path):
        """Safely load model from file"""
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            with open(path, 'rb') as f:
                obj = joblib.load(f)
            logger.info(f"Loaded {os.path.basename(path)} with joblib")
            return obj
        except Exception as e:
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                logger.info(f"Loaded {os.path.basename(path)} with pickle")
                return obj
            except Exception as e:
                logger.error(f"Failed to load {os.path.basename(path)}: {e}")
                return None
    
    def load_models(self):
        """Load ML models"""
        logger.info("Loading models...")
        self.model = self.load_model_safe(Config.MODEL_PATH)
        self.target_encoder = self.load_model_safe(Config.ENCODER_PATH)
        
        if self.model is not None:
            try:
                if hasattr(self.model, 'feature_names_in_'):
                    logger.info(f"Model expects features: {self.model.feature_names_in_}")
                elif hasattr(self.model, 'n_features_in_'):
                    logger.info(f"Model expects {self.model.n_features_in_} features")
            except Exception as e:
                logger.warning(f"Could not determine model features: {e}")
    
    def is_available(self):
        """Check if ML model is available"""
        return self.model is not None and self.target_encoder is not None

model_service = ModelService()

# ============================================================================
# CROP REQUIREMENTS AND FERTILIZER DATA
# ============================================================================

CROP_REQUIREMENTS = {
    'Rice': {'N': 50, 'P': 25, 'K': 25},
    'Wheat': {'N': 45, 'P': 20, 'K': 20},
    'Maize': {'N': 60, 'P': 30, 'K': 30},
    'Cotton': {'N': 55, 'P': 25, 'K': 25},
    'Sugarcane': {'N': 75, 'P': 35, 'K': 35},
    'Pulses': {'N': 25, 'P': 20, 'K': 20},
    'Ground Nuts': {'N': 30, 'P': 20, 'K': 20},
    'Millets': {'N': 35, 'P': 15, 'K': 15}
}

SOIL_FACTORS = {
    'Sandy': 1.2,
    'Loamy': 1.0,
    'Clayey': 0.9,
    'Black': 0.95,
    'Red': 1.1
}

# FIX 1: Added 'NPK' to FERTILIZER_COMPOSITION
FERTILIZER_COMPOSITION = {
    'Urea': {'N': 46, 'P': 0, 'K': 0, 'name': 'Urea', 'description': 'High nitrogen fertilizer for vegetative growth'},
    'DAP': {'N': 18, 'P': 46, 'K': 0, 'name': 'DAP', 'description': 'High phosphorus fertilizer for root development'},
    '14-35-14': {'N': 14, 'P': 35, 'K': 14, 'name': '14-35-14', 'description': 'Balanced NPK with high phosphorus'},
    '28-28': {'N': 28, 'P': 28, 'K': 0, 'name': '28-28', 'description': 'Balanced NP fertilizer'},
    '20-20': {'N': 20, 'P': 20, 'K': 0, 'name': '20-20', 'description': 'Balanced NP fertilizer'},
    '17-17-17': {'N': 17, 'P': 17, 'K': 17, 'name': '17-17-17', 'description': 'Complete balanced NPK fertilizer'},
    'No Fertilizer': {'N': 0, 'P': 0, 'K': 0, 'name': 'No Fertilizer', 'description': 'No fertilizer needed'},
    'NPK': {'N': 20, 'P': 20, 'K': 20, 'name': 'NPK', 'description': 'Balanced NPK fertilizer'}  # ADDED
}

# FIX 2: Added 'NPK' to FERTILIZER_INFO
FERTILIZER_INFO = {
    'Urea': {
        'safety': 'Wear gloves and mask while handling. Store in cool dry place. Keep away from children and animals.',
        'irrigation': 'Apply to moist soil and irrigate immediately. Use 2-3 light irrigations over the next week.',
        'timing': 'Best applied during vegetative growth stage. Split application recommended - 50% at sowing, 50% at tillering.',
        'method': 'Broadcast evenly and incorporate into soil. Apply 2-3 inches deep near root zone.',
        'precautions': 'Do not mix with lime. Avoid contact with seeds. Do not apply on waterlogged soil.',
        'storage': 'Store in airtight containers away from moisture and direct sunlight.'
    },
    'DAP': {
        'safety': 'Use protective equipment. Avoid inhalation of dust. Wash hands thoroughly after use.',
        'irrigation': 'Irrigate immediately after application. Maintain field capacity for 3-4 days.',
        'timing': 'Apply at sowing time or before flowering. Best for root development stage.',
        'method': 'Place 2-3 inches below and to the side of seeds. Band application is most effective.',
        'precautions': 'Do not mix with urea before application. Keep away from moisture during storage.',
        'storage': 'Store in sealed bags in dry conditions. Use within 6 months of manufacture.'
    },
    '14-35-14': {
        'safety': 'Handle with care. Use gloves and mask. Avoid contact with eyes and skin.',
        'irrigation': 'Light irrigation after application. Maintain soil moisture for nutrient uptake.',
        'timing': 'Apply at planting and during early vegetative growth. Ideal for phosphorus-deficient soils.',
        'method': 'Band placement near seed is most effective. Can be applied as foliar spray at low concentrations.',
        'precautions': 'Do not apply on sandy soils without irrigation. Avoid application before heavy rain.',
        'storage': 'Store in cool dry place away from direct sunlight.'
    },
    '28-28': {
        'safety': 'Wear protective gear while handling. Keep away from food and feed.',
        'irrigation': 'Irrigate within 24 hours of application. Ensure uniform moisture distribution.',
        'timing': 'Apply as basal dose at sowing. Can be used for top dressing at tillering stage.',
        'method': 'Broadcast and incorporate into soil. For row crops, apply in bands 2-3 inches from seed.',
        'precautions': 'Do not over-apply. Keep out of reach of children.',
        'storage': 'Store in original packaging in dry, ventilated area.'
    },
    '20-20': {
        'safety': 'Use recommended protective equipment. Avoid creating dust.',
        'irrigation': 'Light irrigation after application. Maintain moisture for nutrient availability.',
        'timing': 'Apply at planting and during active growth stages. Split application recommended.',
        'method': 'Broadcast evenly and mix with soil. For best results, apply in bands near root zone.',
        'precautions': 'Do not apply to frozen or waterlogged soil. Avoid contact with foliage.',
        'storage': 'Store in sealed containers in dry place away from chemicals.'
    },
    '17-17-17': {
        'safety': 'Handle with care. Use gloves and protective clothing. Wash after handling.',
        'irrigation': 'Irrigate immediately after application. Maintain field capacity for nutrient availability.',
        'timing': 'Apply at sowing and as side dressing during vegetative stage. Suitable for all growth stages.',
        'method': 'Broadcast and incorporate into soil. Can be applied through fertigation systems.',
        'precautions': 'Do not mix with lime or calcium-containing materials. Avoid over-application.',
        'storage': 'Store in cool dry place away from moisture and direct sunlight.'
    },
    'No Fertilizer': {
        'safety': 'No fertilizer application needed at this time.',
        'irrigation': 'Follow standard irrigation practices for your crop.',
        'timing': 'No fertilizer application needed. Monitor crop growth regularly.',
        'method': 'No application required. Maintain regular farming practices.',
        'precautions': 'Continue monitoring soil nutrients. Re-test soil after one season.',
        'storage': 'Not applicable.'
    },
    'NPK': {  # ADDED
        'safety': 'Wear gloves and mask while handling. Store in cool dry place.',
        'irrigation': 'Apply to moist soil and irrigate immediately. Maintain soil moisture.',
        'timing': 'Apply during vegetative growth stage. Split application recommended.',
        'method': 'Broadcast evenly and incorporate into soil. Apply 2-3 inches deep.',
        'precautions': 'Do not over-apply. Keep away from water sources.',
        'storage': 'Store in airtight containers away from moisture and direct sunlight.'
    }
}

# ============================================================================
# FERTILIZER OPTIMIZATION ENGINE
# ============================================================================

class OptimizationService:
    """
    Fertilizer Optimization Engine
    Creates optimized split-application fertilizer schedules based on:
    - Crop nutrient requirements
    - Current soil nutrients
    - Fertilizer composition
    - Optimal nutrient balance
    - Split application timing
    """
    
    def __init__(self, db):
        self.db = db
        self.crop_requirements = CROP_REQUIREMENTS
        self.fertilizer_composition = FERTILIZER_COMPOSITION
        self.soil_factors = SOIL_FACTORS
        
        # Stage definitions with timing and purpose
        self.stage_definitions = [
            {
                "stage": "Basal Application",
                "time_gap": "At sowing",
                "purpose": "Root development and early growth",
                "distribution": 0.30,  # 30% of total nutrients
                "nutrient_focus": ["P", "N"]  # Focus on P and some N
            },
            {
                "stage": "Vegetative Stage",
                "time_gap": "20-25 days after sowing",
                "purpose": "Vegetative growth and tillering",
                "distribution": 0.45,  # 45% of total nutrients
                "nutrient_focus": ["N", "K"]  # Focus on N and K
            },
            {
                "stage": "Flowering Stage",
                "time_gap": "40-45 days after sowing",
                "purpose": "Flowering and fruit development",
                "distribution": 0.25,  # 25% of total nutrients
                "nutrient_focus": ["K", "P"]  # Focus on K and P
            }
        ]
        
        # Fertilizer suitability scores for different stages
        self.fertilizer_suitability = {
            "Urea": {
                "Basal Application": 0.6,  # Less suitable for basal
                "Vegetative Stage": 0.95,   # Highly suitable for vegetative
                "Flowering Stage": 0.4       # Less suitable for flowering
            },
            "DAP": {
                "Basal Application": 0.95,   # Highly suitable for basal
                "Vegetative Stage": 0.7,     # Moderately suitable
                "Flowering Stage": 0.5        # Less suitable
            },
            "14-35-14": {
                "Basal Application": 0.9,     # Good for basal (high P)
                "Vegetative Stage": 0.8,      # Good for vegetative
                "Flowering Stage": 0.7         # Moderate for flowering
            },
            "28-28": {
                "Basal Application": 0.8,      # Good for basal
                "Vegetative Stage": 0.85,      # Good for vegetative
                "Flowering Stage": 0.6          # Moderate
            },
            "20-20": {
                "Basal Application": 0.7,       # Moderate
                "Vegetative Stage": 0.8,        # Good
                "Flowering Stage": 0.7           # Moderate
            },
            "17-17-17": {
                "Basal Application": 0.8,        # Good
                "Vegetative Stage": 0.85,        # Good
                "Flowering Stage": 0.85           # Good - balanced
            },
            "No Fertilizer": {
                "Basal Application": 0,
                "Vegetative Stage": 0,
                "Flowering Stage": 0
            },
            "NPK": {  # ADDED
                "Basal Application": 0.8,
                "Vegetative Stage": 0.85,
                "Flowering Stage": 0.8
            }
        }
        
        logger.info("Optimization Service initialized")
    
    def calculate_nutrient_deficit(self, crop, soil_n, soil_p, soil_k):
        """
        Calculate nutrient deficits based on crop requirements
        
        Args:
            crop (str): Crop type
            soil_n (float): Soil nitrogen level (kg/ha)
            soil_p (float): Soil phosphorous level (kg/ha)
            soil_k (float): Soil potassium level (kg/ha)
        
        Returns:
            dict: Nutrient deficits with absolute and percentage values
        """
        if crop not in self.crop_requirements:
            logger.warning(f"Crop {crop} not found in requirements, using default")
            crop = 'Rice'  # Default
        
        req = self.crop_requirements[crop]
        
        # Calculate deficits (ensure non-negative)
        n_deficit = max(0, req['N'] - soil_n)
        p_deficit = max(0, req['P'] - soil_p)
        k_deficit = max(0, req['K'] - soil_k)
        
        # Calculate deficit percentages
        total_required = req['N'] + req['P'] + req['K']
        total_available = soil_n + soil_p + soil_k
        
        # Overall fulfillment percentage
        fulfillment_pct = min(100, (total_available / total_required * 100)) if total_required > 0 else 0
        
        return {
            'N': round(n_deficit, 2),
            'P': round(p_deficit, 2),
            'K': round(k_deficit, 2),
            'N_pct': round((n_deficit / req['N'] * 100) if req['N'] > 0 else 0, 2),
            'P_pct': round((p_deficit / req['P'] * 100) if req['P'] > 0 else 0, 2),
            'K_pct': round((k_deficit / req['K'] * 100) if req['K'] > 0 else 0, 2),
            'fulfillment_pct': round(fulfillment_pct, 2)
        }
    
    def select_optimal_fertilizer_for_stage(self, stage, deficits, available_fertilizers=None):
        """
        Select the optimal fertilizer for a specific growth stage
        
        Args:
            stage (str): Growth stage name
            deficits (dict): Nutrient deficits
            available_fertilizers (list): List of available fertilizers (None = all)
        
        Returns:
            str: Optimal fertilizer name for the stage
        """
        if available_fertilizers is None:
            available_fertilizers = list(self.fertilizer_composition.keys())
        
        # Remove 'No Fertilizer' if there are significant deficits
        if deficits['N'] > 5 or deficits['P'] > 5 or deficits['K'] > 5:
            if 'No Fertilizer' in available_fertilizers:
                available_fertilizers = [f for f in available_fertilizers if f != 'No Fertilizer']
        
        stage_info = next((s for s in self.stage_definitions if s['stage'] == stage), None)
        if not stage_info:
            return '17-17-17'  # Default balanced fertilizer
        
        focus_nutrients = stage_info['nutrient_focus']
        
        best_fertilizer = None
        best_score = -1
        
        for fertilizer in available_fertilizers:
            comp = self.fertilizer_composition.get(fertilizer, {'N': 0, 'P': 0, 'K': 0})
            
            # Calculate nutrient coverage score for this stage
            score = 0
            
            # Focus nutrients get higher weight
            if 'N' in focus_nutrients and comp['N'] > 0:
                # How well does this fertilizer cover N deficit?
                n_coverage = min(1.0, (comp['N'] / 46) * 1.5)  # Normalize to Urea (46% N)
                score += n_coverage * 2.0
            
            if 'P' in focus_nutrients and comp['P'] > 0:
                p_coverage = min(1.0, (comp['P'] / 46) * 1.5)  # Normalize to DAP (46% P)
                score += p_coverage * 2.0
            
            if 'K' in focus_nutrients and comp['K'] > 0:
                k_coverage = min(1.0, (comp['K'] / 17) * 1.5)  # Normalize to 17-17-17 (17% K)
                score += k_coverage * 2.0
            
            # Add stage suitability factor
            suitability = self.fertilizer_suitability.get(fertilizer, {}).get(stage, 0.5)
            score *= (0.5 + suitability)
            
            # Penalize if fertilizer doesn't match deficits
            if deficits['N'] < 5 and comp['N'] > 30:
                score *= 0.7  # Too much N when not needed
            if deficits['P'] < 5 and comp['P'] > 30:
                score *= 0.7  # Too much P when not needed
            if deficits['K'] < 5 and comp['K'] > 15:
                score *= 0.7  # Too much K when not needed
            
            if score > best_score:
                best_score = score
                best_fertilizer = fertilizer
        
        return best_fertilizer if best_fertilizer else '17-17-17'
    
    def calculate_stage_dose(self, stage, fertilizer, total_nutrient_needs, soil_factor=1.0):
        """
        Calculate fertilizer dose for a specific stage
        
        Args:
            stage (dict): Stage definition
            fertilizer (str): Fertilizer name
            total_nutrient_needs (dict): Total nutrient deficits
            soil_factor (float): Soil type adjustment factor
        
        Returns:
            float: Calculated dose for the stage
        """
        comp = self.fertilizer_composition.get(fertilizer, {'N': 0, 'P': 0, 'K': 0})
        
        # Calculate stage nutrient requirements based on distribution
        stage_n_needed = total_nutrient_needs['N'] * stage['distribution']
        stage_p_needed = total_nutrient_needs['P'] * stage['distribution']
        stage_k_needed = total_nutrient_needs['K'] * stage['distribution']
        
        # Calculate doses based on each nutrient
        doses = []
        
        if stage_n_needed > 0 and comp['N'] > 0:
            doses.append((stage_n_needed / comp['N']) * 100)
        if stage_p_needed > 0 and comp['P'] > 0:
            doses.append((stage_p_needed / comp['P']) * 100)
        if stage_k_needed > 0 and comp['K'] > 0:
            doses.append((stage_k_needed / comp['K']) * 100)
        
        if not doses:
            return 0
        
        # Take the maximum required dose to meet all nutrient needs
        base_dose = max(doses)
        
        # Apply soil factor
        final_dose = base_dose * soil_factor
        
        # Cap at reasonable maximum per stage (typically not more than 100 kg/acre per application)
        return min(final_dose, 100)
    
    def calculate_nutrient_supply(self, stages):
        """
        Calculate total nutrients supplied by the optimized schedule
        
        Args:
            stages (list): List of stage dictionaries with fertilizer and dose
        
        Returns:
            dict: Total N, P, K supplied
        """
        total_n = 0
        total_p = 0
        total_k = 0
        
        for stage in stages:
            fertilizer = stage['fertilizer']
            dose = stage['dose']
            
            comp = self.fertilizer_composition.get(fertilizer, {'N': 0, 'P': 0, 'K': 0})
            
            # Calculate nutrients supplied (kg/ha)
            # Dose is in kg/acre, convert to kg/ha (multiply by 2.471)
            dose_ha = dose * 2.471
            
            total_n += (dose_ha * comp['N'] / 100)
            total_p += (dose_ha * comp['P'] / 100)
            total_k += (dose_ha * comp['K'] / 100)
        
        return {
            'nitrogen': round(total_n, 2),
            'phosphorous': round(total_p, 2),
            'potassium': round(total_k, 2)
        }
    
    # FIX 3: Fixed division by zero in calculate_optimization_score
    def calculate_optimization_score(self, deficits, nutrient_supply, crop):
        """
        Calculate optimization score based on how well the plan meets crop needs
        
        Args:
            deficits (dict): Original nutrient deficits
            nutrient_supply (dict): Total nutrients supplied by the plan
            crop (str): Crop type
        
        Returns:
            int: Optimization score (0-100)
        """
        if crop not in self.crop_requirements:
            crop = 'Rice'
        
        req = self.crop_requirements[crop]
        
        # Calculate how well we meet the requirements with safe division
        # Handle N
        if deficits['N'] > 0:
            n_ratio = min(1.0, nutrient_supply['nitrogen'] / deficits['N'])
            # Penalize over-application (more than 20% over)
            if nutrient_supply['nitrogen'] > deficits['N'] * 1.2:
                n_ratio = max(0, 1.0 - (nutrient_supply['nitrogen'] - deficits['N'] * 1.2) / deficits['N'])
        else:
            # If no deficit, check if we're adding unnecessary nitrogen
            n_ratio = 1.0 if nutrient_supply['nitrogen'] <= 5 else max(0, 1.0 - (nutrient_supply['nitrogen'] / 50))
        
        # Handle P
        if deficits['P'] > 0:
            p_ratio = min(1.0, nutrient_supply['phosphorous'] / deficits['P'])
            # Penalize over-application (more than 20% over)
            if nutrient_supply['phosphorous'] > deficits['P'] * 1.2:
                p_ratio = max(0, 1.0 - (nutrient_supply['phosphorous'] - deficits['P'] * 1.2) / deficits['P'])
        else:
            # If no deficit, check if we're adding unnecessary phosphorus
            p_ratio = 1.0 if nutrient_supply['phosphorous'] <= 5 else max(0, 1.0 - (nutrient_supply['phosphorous'] / 50))
        
        # Handle K
        if deficits['K'] > 0:
            k_ratio = min(1.0, nutrient_supply['potassium'] / deficits['K'])
            # Penalize over-application (more than 20% over)
            if nutrient_supply['potassium'] > deficits['K'] * 1.2:
                k_ratio = max(0, 1.0 - (nutrient_supply['potassium'] - deficits['K'] * 1.2) / deficits['K'])
        else:
            # If no deficit, check if we're adding unnecessary potassium
            k_ratio = 1.0 if nutrient_supply['potassium'] <= 5 else max(0, 1.0 - (nutrient_supply['potassium'] / 50))
        
        # Calculate weighted score (N is often most important)
        weighted_score = (n_ratio * 0.4 + p_ratio * 0.35 + k_ratio * 0.25) * 100
        
        return int(weighted_score)
    
    def generate_irrigation_recommendation(self, stages, crop, soil_type):
        """
        Generate irrigation recommendations based on the fertilizer schedule
        
        Args:
            stages (list): Optimized fertilizer stages
            crop (str): Crop type
            soil_type (str): Soil type
        
        Returns:
            dict: Irrigation recommendations
        """
        soil_texture = soil_type.lower()
        
        # Base irrigation intervals based on soil type
        if 'sandy' in soil_texture:
            base_interval = "3-4 days"
            water_amount = "light and frequent"
        elif 'clay' in soil_texture:
            base_interval = "7-10 days"
            water_amount = "deep and infrequent"
        elif 'loam' in soil_texture:
            base_interval = "5-7 days"
            water_amount = "moderate"
        else:
            base_interval = "5-7 days"
            water_amount = "as needed"
        
        recommendations = []
        
        for stage in stages:
            if stage['stage'] == 'Basal Application':
                recommendations.append({
                    'stage': stage['stage'],
                    'timing': "Immediately after fertilizer application",
                    'instruction': f"Apply {water_amount} irrigation to incorporate fertilizer into soil. Maintain field capacity for 2-3 days."
                })
            elif stage['stage'] == 'Vegetative Stage':
                recommendations.append({
                    'stage': stage['stage'],
                    'timing': f"Starting {stage['time_gap']}",
                    'instruction': f"Maintain soil moisture at 80% field capacity. Irrigate every {base_interval} depending on rainfall."
                })
            elif stage['stage'] == 'Flowering Stage':
                recommendations.append({
                    'stage': stage['stage'],
                    'timing': f"Starting {stage['time_gap']}",
                    'instruction': "Critical stage - avoid water stress. Maintain adequate moisture. Reduce irrigation 10-15 days before harvest."
                })
        
        return {
            'base_recommendation': f"For {soil_type} soil, use {water_amount} irrigation every {base_interval}.",
            'stage_specific': recommendations
        }
    
    def generate_fertilizer_schedule(self, crop, soil_type, nitrogen, phosphorous, potassium, 
                                     predicted_fertilizer, base_dose):
        """
        Generate an optimized fertilizer application schedule
        
        Args:
            crop (str): Crop type
            soil_type (str): Soil type
            nitrogen (float): Soil nitrogen level (kg/ha)
            phosphorous (float): Soil phosphorous level (kg/ha)
            potassium (float): Soil potassium level (kg/ha)
            predicted_fertilizer (str): Primary fertilizer recommended
            base_dose (float): Base dose calculated (kg/acre)
        
        Returns:
            dict: Complete optimization result with stages, nutrient supply, score, etc.
        """
        logger.info(f"Generating optimized fertilizer schedule for {crop} on {soil_type} soil")
        
        # Check cache first
        cached = self.db.get_optimization_cache(crop, soil_type, nitrogen, phosphorous, potassium)
        if cached:
            logger.info(f"Returning cached optimization for {crop}")
            return cached
        
        # Calculate nutrient deficits
        deficits = self.calculate_nutrient_deficit(crop, nitrogen, phosphorous, potassium)
        
        # If no significant deficits, return simple schedule
        if deficits['N'] < 5 and deficits['P'] < 5 and deficits['K'] < 5:
            result = {
                "stages": [
                    {
                        "stage": "Basal Application",
                        "fertilizer": "No Fertilizer",
                        "dose": 0,
                        "time_gap": "At sowing",
                        "purpose": "Soil nutrients are sufficient - no fertilizer needed"
                    }
                ],
                "total_nutrient_supply": {
                    "nitrogen": 0,
                    "phosphorous": 0,
                    "potassium": 0
                },
                "optimization_score": 100,
                "soil_nutrient_balance": "Optimal",
                "irrigation_recommendation": {
                    "base_recommendation": f"For {soil_type} soil, follow standard irrigation practices for {crop}.",
                    "stage_specific": []
                },
                "summary": f"Your soil already has sufficient nutrients for {crop}. No fertilizer application needed at this time."
            }
            
            # Cache the result
            self.db.save_optimization(crop, soil_type, nitrogen, phosphorous, potassium,
                                     predicted_fertilizer, base_dose, result)
            
            return result
        
        # Soil factor for dose adjustment
        soil_factor = self.soil_factors.get(soil_type, 1.0)
        
        # Available fertilizers (start with all, but prioritize predicted fertilizer)
        all_fertilizers = list(self.fertilizer_composition.keys())
        
        # Ensure predicted fertilizer is available
        if predicted_fertilizer not in all_fertilizers:
            logger.warning(f"Predicted fertilizer {predicted_fertilizer} not found, using 17-17-17")
            predicted_fertilizer = '17-17-17'
        
        # Generate stages
        stages = []
        
        for stage_def in self.stage_definitions:
            stage_name = stage_def['stage']
            
            # Select optimal fertilizer for this stage
            # Try to use predicted fertilizer if suitable
            if stage_name in self.fertilizer_suitability.get(predicted_fertilizer, {}):
                suitability = self.fertilizer_suitability[predicted_fertilizer][stage_name]
                if suitability > 0.6:  # If reasonably suitable, use it
                    fertilizer = predicted_fertilizer
                else:
                    # Otherwise select best available
                    fertilizer = self.select_optimal_fertilizer_for_stage(
                        stage_name, deficits, all_fertilizers
                    )
            else:
                fertilizer = self.select_optimal_fertilizer_for_stage(
                    stage_name, deficits, all_fertilizers
                )
            
            # Calculate dose for this stage
            dose = self.calculate_stage_dose(stage_def, fertilizer, deficits, soil_factor)
            
            # Only include stages with meaningful dose
            if dose > 5:
                stages.append({
                    "stage": stage_name,
                    "fertilizer": fertilizer,
                    "dose": round(dose, 2),
                    "time_gap": stage_def['time_gap'],
                    "purpose": stage_def['purpose']
                })
        
        # Ensure we have at least one stage
        if not stages:
            # Fallback to single application
            stages.append({
                "stage": "Single Application",
                "fertilizer": predicted_fertilizer,
                "dose": round(base_dose * soil_factor, 2),
                "time_gap": "At sowing",
                "purpose": "Complete fertilizer application"
            })
        
        # Calculate total nutrients supplied
        nutrient_supply = self.calculate_nutrient_supply(stages)
        
        # Calculate optimization score
        optimization_score = self.calculate_optimization_score(deficits, nutrient_supply, crop)
        
        # Determine soil nutrient balance
        if deficits['fulfillment_pct'] >= 90:
            balance = "Optimal"
        elif deficits['fulfillment_pct'] >= 70:
            balance = "Good"
        elif deficits['fulfillment_pct'] >= 50:
            balance = "Moderate"
        else:
            balance = "Needs Improvement"
        
        # Generate irrigation recommendation
        irrigation_rec = self.generate_irrigation_recommendation(stages, crop, soil_type)
        
        # Create summary
        if optimization_score >= 90:
            quality = "excellent"
        elif optimization_score >= 75:
            quality = "good"
        elif optimization_score >= 60:
            quality = "moderate"
        else:
            quality = "basic"
        
        summary = (f"Optimized {quality} fertilizer schedule for {crop} on {soil_type} soil. "
                  f"Total nutrients supplied: N: {nutrient_supply['nitrogen']} kg/ha, "
                  f"P: {nutrient_supply['phosphorous']} kg/ha, "
                  f"K: {nutrient_supply['potassium']} kg/ha. "
                  f"This schedule improves nutrient use efficiency by splitting applications "
                  f"according to crop growth stages.")
        
        result = {
            "stages": stages,
            "total_nutrient_supply": nutrient_supply,
            "optimization_score": optimization_score,
            "soil_nutrient_balance": balance,
            "irrigation_recommendation": irrigation_rec,
            "summary": summary
        }
        
        # Cache the result
        self.db.save_optimization(crop, soil_type, nitrogen, phosphorous, potassium,
                                 predicted_fertilizer, base_dose, result)
        
        logger.info(f"Optimization complete for {crop} with score {optimization_score}")
        
        return result

# ============================================================================
# PREDICTION SERVICE
# ============================================================================

class PredictionService:
    """Prediction service"""
    
    def __init__(self, model_service, db):
        self.model_service = model_service
        self.db = db
    
    def check_sufficient_nutrients(self, crop, n, p, k):
        """Check if soil has sufficient nutrients for the crop"""
        if crop not in CROP_REQUIREMENTS:
            return False
        req = CROP_REQUIREMENTS[crop]
        
        n_sufficient = n >= req['N'] * 0.9
        p_sufficient = p >= req['P'] * 0.9
        k_sufficient = k >= req['K'] * 0.9
        
        return n_sufficient and p_sufficient and k_sufficient
    
    def rule_based_prediction(self, crop_type, n, p, k):
        """Rule-based fertilizer recommendation"""
        if crop_type not in CROP_REQUIREMENTS:
            return '17-17-17'
        
        req = CROP_REQUIREMENTS[crop_type]
        
        n_deficit = max(0, req['N'] - n)
        p_deficit = max(0, req['P'] - p)
        k_deficit = max(0, req['K'] - k)
        
        if n_deficit <= 5 and p_deficit <= 5 and k_deficit <= 5:
            return 'No Fertilizer'
        
        deficits = {'N': n_deficit, 'P': p_deficit, 'K': k_deficit}
        most_deficient = max(deficits, key=deficits.get)
        
        if most_deficient == 'N' and n_deficit > 15:
            return 'Urea'
        elif most_deficient == 'P' and p_deficit > 15:
            return 'DAP' if n_deficit < 10 else '14-35-14'
        elif most_deficient == 'K' and k_deficit > 15:
            if n_deficit > 15 and p_deficit > 15:
                return '17-17-17'
            elif n_deficit > 15:
                return '20-20'
            else:
                return '28-28'
        elif n_deficit > 15 and p_deficit > 15:
            return '20-20'
        elif n_deficit > 15 and k_deficit > 15:
            return '17-17-17'
        elif p_deficit > 15 and k_deficit > 15:
            return '14-35-14'
        else:
            return '17-17-17'
    
    def make_prediction(self, temperature, humidity, moisture, soil_type, crop_type, 
                       nitrogen, phosphorous, potassium):
        """Make fertilizer prediction using ML model or rule-based system"""
        
        if self.check_sufficient_nutrients(crop_type, nitrogen, phosphorous, potassium):
            return 'No Fertilizer', 95.0
        
        if self.model_service.is_available():
            try:
                # Create DataFrame with correct column names
                input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, 
                                          nitrogen, phosphorous, potassium]],
                                        columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type',
                                               'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium'])
                
                logger.info(f"ML prediction input: {input_data.to_dict()}")
                
                prediction = self.model_service.model.predict(input_data)[0]
                confidence = 85.0
                
                if hasattr(self.model_service.model, 'predict_proba'):
                    try:
                        probabilities = self.model_service.model.predict_proba(input_data)[0]
                        confidence = max(probabilities) * 100
                    except Exception as e:
                        logger.warning(f"Could not get prediction probabilities: {e}")
                
                if hasattr(self.model_service.target_encoder, 'inverse_transform'):
                    fertilizer = self.model_service.target_encoder.inverse_transform([prediction])[0]
                else:
                    fertilizer = prediction
                
                logger.info(f"ML prediction successful: {fertilizer} with {confidence:.2f}% confidence")
                return fertilizer, round(confidence, 2)
                
            except Exception as e:
                logger.error(f"ML prediction error: {e}", exc_info=True)
                # Fall back to rule-based system
        
        logger.info("Using rule-based prediction")
        fertilizer = self.rule_based_prediction(crop_type, nitrogen, phosphorous, potassium)
        return fertilizer, 75.0
    
    def calculate_dynamic_dose(self, crop, soil_type, soil_n, soil_p, soil_k, fertilizer_name):
        """Calculate dynamic fertilizer dose based on crop requirements and soil nutrients"""
        if crop not in CROP_REQUIREMENTS:
            crop = 'Rice'
        
        req = CROP_REQUIREMENTS[crop]
        
        n_req = req['N']
        p_req = req['P']
        k_req = req['K']
        
        n_deficit = max(0, n_req - soil_n)
        p_deficit = max(0, p_req - soil_p)
        k_deficit = max(0, k_req - soil_k)
        
        if fertilizer_name == 'No Fertilizer':
            return {
                'dose': 0,
                'n_deficit': 0,
                'p_deficit': 0,
                'k_deficit': 0,
                'n_deficit_amount': 0,
                'p_deficit_amount': 0,
                'k_deficit_amount': 0,
                'fulfilled_pct': 100,
                'explanation': f"Your soil already has enough nutrients for {crop}. No fertilizer needed.",
                'base_dose': 0,
                'soil_factor': SOIL_FACTORS.get(soil_type, 1.0),
                'n_content': 0,
                'p_content': 0,
                'k_content': 0
            }
        
        # FIX 4: Safe fallback for unknown fertilizer
        comp = FERTILIZER_COMPOSITION.get(fertilizer_name, {'N': 20, 'P': 20, 'K': 20})
        
        doses = []
        if n_deficit > 0 and comp['N'] > 0:
            doses.append((n_deficit / comp['N']) * 100)
        if p_deficit > 0 and comp['P'] > 0:
            doses.append((p_deficit / comp['P']) * 100)
        if k_deficit > 0 and comp['K'] > 0:
            doses.append((k_deficit / comp['K']) * 100)
        
        base_dose = max(doses) if doses else 0
        soil_factor = SOIL_FACTORS.get(soil_type, 1.0)
        final_dose = base_dose * soil_factor
        final_dose = min(final_dose, 200)
        
        n_deficit_pct = round(max(0, n_req - soil_n) / n_req * 100, 2) if n_req > 0 else 0
        p_deficit_pct = round(max(0, p_req - soil_p) / p_req * 100, 2) if p_req > 0 else 0
        k_deficit_pct = round(max(0, k_req - soil_k) / k_req * 100, 2) if k_req > 0 else 0
        
        n_deficit_amount = round(max(0, n_req - soil_n), 2)
        p_deficit_amount = round(max(0, p_req - soil_p), 2)
        k_deficit_amount = round(max(0, k_req - soil_k), 2)
        
        total_required = n_req + p_req + k_req
        total_available = soil_n + soil_p + soil_k
        fulfilled_pct = round(min(100, (total_available / total_required * 100)), 2) if total_required > 0 else 0
        
        explanation = f"For your {crop} crop, "
        if n_deficit > 15:
            explanation += f"nitrogen is low ({round(soil_n)} kg/ha). "
        elif n_deficit > 5:
            explanation += f"nitrogen is moderate ({round(soil_n)} kg/ha). "
        else:
            explanation += f"nitrogen is good ({round(soil_n)} kg/ha). "
        
        if p_deficit > 15:
            explanation += f"Phosphorous is low ({round(soil_p)} kg/ha). "
        elif p_deficit > 5:
            explanation += f"Phosphorous is moderate ({round(soil_p)} kg/ha). "
        else:
            explanation += f"Phosphorous is good ({round(soil_p)} kg/ha). "
        
        if k_deficit > 15:
            explanation += f"Potassium is low ({round(soil_k)} kg/ha). "
        elif k_deficit > 5:
            explanation += f"Potassium is moderate ({round(soil_k)} kg/ha). "
        else:
            explanation += f"Potassium is good ({round(soil_k)} kg/ha). "
        
        explanation += f"That's why we recommend {fertilizer_name}. This fertilizer contains {comp['N']}% Nitrogen, {comp['P']}% Phosphorous, and {comp['K']}% Potassium."
        
        return {
            'dose': round(final_dose, 2),
            'n_deficit': n_deficit_pct,
            'p_deficit': p_deficit_pct,
            'k_deficit': k_deficit_pct,
            'n_deficit_amount': n_deficit_amount,
            'p_deficit_amount': p_deficit_amount,
            'k_deficit_amount': k_deficit_amount,
            'fulfilled_pct': fulfilled_pct,
            'explanation': explanation,
            'base_dose': round(base_dose, 2),
            'soil_factor': soil_factor,
            'n_content': comp['N'],
            'p_content': comp['P'],
            'k_content': comp['K']
        }

# Initialize prediction service
prediction_service = PredictionService(model_service, db)

# ============================================================================
# WEATHER SERVICE
# ============================================================================

class WeatherService:
    """Weather service with caching"""
    
    def __init__(self, db, api_key):
        self.db = db
        self.api_key = api_key
    
    def calculate_uv_index(self, clouds):
        """Calculate UV index based on cloud cover and time"""
        hour = datetime.now().hour
        if hour < 8 or hour > 17:
            return 0
        base_uv = 8
        cloud_factor = 1 - (clouds / 100) * 0.7
        time_factor = 1 - abs(hour - 12) / 6 * 0.5
        return base_uv * cloud_factor * time_factor
    
    def fetch_by_city(self, city):
        """Fetch weather data by city name with caching"""
        if not self.api_key:
            logger.error("OpenWeather API key not configured")
            return {'error': 'Weather API not configured', 'status_code': 500}, 500
        
        try:
            # Check cache first
            cached = self.db.get_weather_cache(city)
            if cached:
                logger.info(f"Returning cached weather data for {city}")
                return cached
            
            # Fetch from API
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Weather API returned {response.status_code} for city {city}")
                return {'error': 'Weather API failed', 'status_code': response.status_code}, response.status_code
            
            current = response.json()
            
            weather_data = {
                'city': current['name'],
                'country': current['sys']['country'],
                'temperature': round(current['main']['temp'], 1),
                'feels_like': round(current['main']['feels_like'], 1),
                'humidity': current['main']['humidity'],
                'pressure': current['main']['pressure'],
                'description': current['weather'][0]['description'],
                'wind_speed': round(current['wind']['speed'] * 3.6, 1),
                'wind_direction': current['wind'].get('deg', 0),
                'clouds': current['clouds']['all'],
                'sunrise': datetime.fromtimestamp(current['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(current['sys']['sunset']).strftime('%H:%M'),
                'uv_index': round(self.calculate_uv_index(current['clouds']['all']), 1),
                'air_quality': 'Good'
            }
            
            # Cache the result
            self.db.save_weather_cache(city, weather_data['country'], weather_data)
            
            logger.info(f"Weather data fetched and cached for {city}")
            return weather_data
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching weather for {city}")
            return {'error': 'Weather API timeout', 'status_code': 504}, 504
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching weather for {city}: {e}")
            return {'error': 'Weather API request failed', 'status_code': 500}, 500
        except Exception as e:
            logger.error(f"Unexpected error fetching weather for {city}: {e}")
            return {'error': 'Weather service error', 'status_code': 500}, 500
    
    def fetch_by_coords(self, lat, lon):
        """Fetch weather data by coordinates"""
        if not self.api_key:
            logger.error("OpenWeather API key not configured")
            return {'error': 'Weather API not configured', 'status_code': 500}, 500
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Weather API returned {response.status_code} for coordinates {lat},{lon}")
                return {'error': 'Weather API failed', 'status_code': response.status_code}, response.status_code
            
            current = response.json()
            city = current.get('name', 'Current Location')
            return self.fetch_by_city(city)
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching weather for coordinates {lat},{lon}")
            return {'error': 'Weather API timeout', 'status_code': 504}, 504
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching weather for coordinates {lat},{lon}: {e}")
            return {'error': 'Weather API request failed', 'status_code': 500}, 500
        except Exception as e:
            logger.error(f"Unexpected error fetching weather for coordinates {lat},{lon}: {e}")
            return {'error': 'Weather service error', 'status_code': 500}, 500

weather_service = WeatherService(db, Config.OPENWEATHER_API_KEY)

# ============================================================================
# PDF SERVICE
# ============================================================================

class PDFService:
    """PDF generation service with multilingual support"""
    
    def __init__(self):
        self.fonts_registered = False
        self.available_fonts = {}  # Track available fonts
        self.register_fonts()
    
    def register_fonts(self):
        """Register fonts for multilingual support with fallbacks"""
        try:
            # Try to register Noto fonts if they exist
            fonts = {
                'en': ('NotoSans-Regular.ttf', 'NotoSans'),
                'hi': ('NotoSansDevanagari-Regular.ttf', 'NotoSansDevanagari'),
                'te': ('NotoSansTelugu-Regular.ttf', 'NotoSansTelugu'),
                'ta': ('NotoSansTamil-Regular.ttf', 'NotoSansTamil'),
                'kn': ('NotoSansKannada-Regular.ttf', 'NotoSansKannada')
            }
            
            # Default fallback fonts
            default_fonts = {
                'en': 'Helvetica',
                'hi': 'Helvetica',
                'te': 'Helvetica',
                'ta': 'Helvetica',
                'kn': 'Helvetica'
            }
            
            for lang, (filename, fontname) in fonts.items():
                font_path = os.path.join(Config.FONTS_DIR, filename)
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont(fontname, font_path))
                        self.available_fonts[lang] = fontname
                        logger.info(f"Registered font {fontname} for {lang}")
                    except Exception as e:
                        logger.warning(f"Failed to register font {fontname}: {e}")
                        self.available_fonts[lang] = default_fonts[lang]
                else:
                    logger.warning(f"Font file not found: {font_path}, using fallback for {lang}")
                    self.available_fonts[lang] = default_fonts[lang]
            
            self.fonts_registered = True
            
        except Exception as e:
            logger.error(f"Failed to register fonts: {e}")
            self.fonts_registered = False
            # Set all to Helvetica as ultimate fallback
            for lang in ['en', 'hi', 'te', 'ta', 'kn']:
                self.available_fonts[lang] = 'Helvetica'
    
    def get_font_name(self, language):
        """Get appropriate font name for language with fallback"""
        if not self.fonts_registered or language not in self.available_fonts:
            return 'Helvetica'
        
        return self.available_fonts.get(language, 'Helvetica')
    
    def get_translation(self, language, key):
        """Get translated text"""
        return TRANSLATIONS.get(language, TRANSLATIONS['en']).get(key, key)
    
    def generate_report(self, data, language='en'):
        """Generate PDF report with professional styling"""
        buffer = io.BytesIO()
        
        # Create document with margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get translations
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        
        # Get font for this language (with fallback)
        font_name = self.get_font_name(language)
        
        # Create styles with fallback font
        styles = getSampleStyleSheet()
        
        # Define styles with the available font
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=28,
            textColor=colors.HexColor('#1e3c72'),
            alignment=TA_CENTER,
            spaceAfter=10,
            leading=34,
            textTransform='uppercase'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=14,
            textColor=colors.HexColor('#2a5298'),
            alignment=TA_CENTER,
            spaceAfter=30,
            leading=18,
            fontStyle='italic'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=16,
            textColor=colors.HexColor('#1e3c72'),
            spaceAfter=12,
            spaceBefore=20,
            leading=20,
            borderWidth=0,
            borderColor=colors.HexColor('#2a5298'),
            borderRadius=5,
            backColor=colors.HexColor('#f0f7fa'),
            leftIndent=10,
            rightIndent=10
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            leading=14,
            spaceAfter=6
        )
        
        label_style = ParagraphStyle(
            'LabelStyle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            leading=14,
            alignment=TA_RIGHT
        )
        
        value_style = ParagraphStyle(
            'ValueStyle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            textColor=colors.HexColor('#1e3c72'),
            leading=14,
            fontWeight='bold'
        )
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
            leading=12
        )
        
        story = []
        
        # Header with gradient effect
        story.append(Paragraph(t.get('report_title', 'Fertilizer Recommendation Report'), title_style))
        story.append(Paragraph(t.get('report_subtitle', 'AI-Powered Precision Farming'), subtitle_style))
        story.append(Spacer(1, 20))
        
        # Date and Location
        date_text = f"<b>{t.get('generated_on', 'Generated on')}:</b> {data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))}"
        location_text = f"<b>{t.get('location', 'Location')}:</b> {data.get('location', 'N/A')}"
        
        story.append(Paragraph(date_text, normal_style))
        story.append(Paragraph(location_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Weather Conditions Section
        story.append(Paragraph(t.get('weather_conditions', 'Weather Conditions'), heading_style))
        
        weather_data = [
            [t.get('temperature', 'Temperature'), f"{data.get('temperature', 0)}°C"],
            [t.get('humidity', 'Humidity'), f"{data.get('humidity', 0)}%"],
            [t.get('moisture', 'Soil Moisture'), f"{data.get('moisture', 0)}%"],
            [t.get('weather_condition', 'Condition'), data.get('weather_condition', 'N/A')]
        ]
        
        weather_table = Table(weather_data, colWidths=[200, 200])
        weather_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f7fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2a5298')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT')
        ]))
        story.append(weather_table)
        story.append(Spacer(1, 20))
        
        # Soil Analysis Section
        story.append(Paragraph(t.get('soil_analysis', 'Soil Analysis'), heading_style))
        
        req = CROP_REQUIREMENTS.get(data.get('crop', ''), {'N': 0, 'P': 0, 'K': 0})
        
        soil_data = [
            [t.get('soil_type', 'Soil Type'), data.get('soil', 'N/A'), ''],
            [t.get('crop_type', 'Crop Type'), data.get('crop', 'N/A'), ''],
            [f"{t.get('nitrogen', 'Nitrogen')} ({t.get('current', 'Current')})", f"{data.get('nitrogen', 0)} kg/ha", f"{t.get('required', 'Required')}: {req['N']} kg/ha"],
            [f"{t.get('phosphorous', 'Phosphorous')} ({t.get('current', 'Current')})", f"{data.get('phosphorous', 0)} kg/ha", f"{t.get('required', 'Required')}: {req['P']} kg/ha"],
            [f"{t.get('potassium', 'Potassium')} ({t.get('current', 'Current')})", f"{data.get('potassium', 0)} kg/ha", f"{t.get('required', 'Required')}: {req['K']} kg/ha"]
        ]
        
        soil_table = Table(soil_data, colWidths=[150, 125, 125])
        soil_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f7fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2a5298')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT')
        ]))
        story.append(soil_table)
        story.append(Spacer(1, 20))
        
        # AI Recommendation Section with colored background
        story.append(Paragraph(t.get('ai_recommendation', 'AI Recommendation'), heading_style))
        
        rec_data = [
            [t.get('fertilizer', 'Fertilizer'), data.get('fertilizer', 'N/A')],
            [t.get('confidence', 'Confidence'), f"{data.get('confidence', 0)}%"],
            [t.get('dose_per_acre', 'Dose per acre'), f"{data.get('dose', 0)} kg"],
            ['Composition', f"N: {data.get('n_content', 0)}%, P: {data.get('p_content', 0)}%, K: {data.get('k_content', 0)}%"]
        ]
        
        rec_table = Table(rec_data, colWidths=[150, 250])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e6f3e6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4CAF50')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT')
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 20))
        
        # Nutrient Deficiency Section
        story.append(Paragraph(t.get('deficiency_analysis', 'Nutrient Deficiency Analysis'), heading_style))
        
        deficiency_data = [
            [t.get('nitrogen_deficit', 'Nitrogen'), f"{data.get('n_deficit', 0)}%", f"{data.get('n_deficit_amount', 0)} kg/ha"],
            [t.get('phosphorous_deficit', 'Phosphorous'), f"{data.get('p_deficit', 0)}%", f"{data.get('p_deficit_amount', 0)} kg/ha"],
            [t.get('potassium_deficit', 'Potassium'), f"{data.get('k_deficit', 0)}%", f"{data.get('k_deficit_amount', 0)} kg/ha"]
        ]
        
        deficiency_table = Table(deficiency_data, colWidths=[150, 100, 150])
        deficiency_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff4e6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ff9800')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT')
        ]))
        story.append(deficiency_table)
        story.append(Spacer(1, 20))
        
        # Requirement Fulfilled
        story.append(Paragraph(t.get('requirement_fulfilled', 'Requirement Fulfilled'), heading_style))
        story.append(Paragraph(f"{data.get('fulfilled_pct', 0)}%", ParagraphStyle('BigNumber', parent=styles['Heading1'], fontName=font_name, fontSize=24, textColor=colors.HexColor('#4CAF50'), alignment=TA_CENTER)))
        story.append(Spacer(1, 20))
        
        # Explanation
        story.append(Paragraph(t.get('explanation', 'Explanation'), heading_style))
        story.append(Paragraph(data.get('explanation', 'No explanation available.'), normal_style))
        story.append(Spacer(1, 20))
        
        # Application Instructions
        story.append(Paragraph(t.get('application_instructions', 'Application Instructions'), heading_style))
        
        instructions = [
            [t.get('safety_precautions', 'Safety Precautions'), data.get('safety', '')],
            [t.get('irrigation_advice', 'Irrigation Advice'), data.get('irrigation', '')],
            [t.get('application_timing', 'Application Timing'), data.get('timing', '')],
            [t.get('application_method', 'Application Method'), data.get('method', '')],
            [t.get('storage_instructions', 'Storage Instructions'), data.get('storage', '')]
        ]
        
        instructions_table = Table(instructions, colWidths=[150, 250])
        instructions_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff4e6')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ff9800')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT')
        ]))
        story.append(instructions_table)
        story.append(Spacer(1, 30))
        
        # Footer
        story.append(Paragraph(t.get('footer_text', 'Smart Agriculture AI Platform - Precision Farming for a Sustainable Future'), footer_style))
        story.append(Paragraph(f"Page 1", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_history_pdf(self, history_data, language='en'):
        """Generate history PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        font_name = self.get_font_name(language)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=20,
            textColor=colors.HexColor('#2a5298'),
            alignment=TA_CENTER
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            leading=12
        )
        
        story = []
        
        story.append(Paragraph(t.get('history_title', 'Prediction History'), title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"{t.get('generated_on', 'Generated on')}: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Spacer(1, 20))
        
        if history_data:
            # Table headers
            data = [[
                t.get('date', 'Date'), 
                t.get('crop_type', 'Crop'), 
                t.get('soil_type', 'Soil'), 
                t.get('fertilizer', 'Fertilizer'), 
                t.get('dose', 'Dose'), 
                t.get('confidence', 'Confidence')
            ]]
            
            # Add rows
            for record in history_data:
                data.append([
                    record.get('date', ''),
                    record.get('crop', ''),
                    record.get('soil', ''),
                    record.get('fertilizer', ''),
                    str(record.get('dose', 0)),
                    f"{record.get('confidence', 0)}%"
                ])
            
            table = Table(data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.5*inch, 1.0*inch, 1.0*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), font_name if font_name != 'Helvetica' else 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No prediction history available.", normal_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

pdf_service = PDFService()

# ============================================================================
# VALIDATION SERVICE
# ============================================================================

class ValidationService:
    """Request validation service with multilingual error responses"""
    
    def __init__(self):
        self.soil_types = Config.SOIL_TYPES
        self.crop_types = Config.CROP_TYPES
    
    def get_translation(self, language, key):
        """Get translated error message"""
        return TRANSLATIONS.get(language, TRANSLATIONS['en']).get(key, key)
    
    def validate_prediction_request(self, form_data, language='en'):
        """Validate prediction request data and return structured errors"""
        errors = []
        
        # Check required fields
        required_fields = ['temperature', 'humidity', 'moisture', 'soil_type', 
                          'crop_type', 'nitrogen', 'phosphorous', 'potassium']
        
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                errors.append({
                    'field': field,
                    'code': f'MISSING_{field.upper()}',
                    'message': self.get_translation(language, f'error_{field}_required')
                })
        
        if errors:
            return False, errors
        
        # Validate temperature
        try:
            temp = float(form_data['temperature'])
            if temp < Config.TEMP_MIN or temp > Config.TEMP_MAX:
                errors.append({
                    'field': 'temperature',
                    'code': 'INVALID_TEMPERATURE',
                    'message': self.get_translation(language, 'error_temperature_range')
                })
        except ValueError:
            errors.append({
                'field': 'temperature',
                'code': 'INVALID_TEMPERATURE',
                'message': self.get_translation(language, 'error_temperature_required')
            })
        
        # Validate humidity
        try:
            humidity = float(form_data['humidity'])
            if humidity < Config.HUMIDITY_MIN or humidity > Config.HUMIDITY_MAX:
                errors.append({
                    'field': 'humidity',
                    'code': 'INVALID_HUMIDITY',
                    'message': self.get_translation(language, 'error_humidity_range')
                })
        except ValueError:
            errors.append({
                'field': 'humidity',
                'code': 'INVALID_HUMIDITY',
                'message': self.get_translation(language, 'error_humidity_required')
            })
        
        # Validate moisture
        try:
            moisture = float(form_data['moisture'])
            if moisture < Config.MOISTURE_MIN or moisture > Config.MOISTURE_MAX:
                errors.append({
                    'field': 'moisture',
                    'code': 'INVALID_MOISTURE',
                    'message': self.get_translation(language, 'error_moisture_range')
                })
        except ValueError:
            errors.append({
                'field': 'moisture',
                'code': 'INVALID_MOISTURE',
                'message': self.get_translation(language, 'error_moisture_required')
            })
        
        # Validate soil type
        if form_data['soil_type'] not in self.soil_types:
            errors.append({
                'field': 'soil_type',
                'code': 'INVALID_SOIL_TYPE',
                'message': self.get_translation(language, 'error_invalid_soil')
            })
        
        # Validate crop type
        if form_data['crop_type'] not in self.crop_types:
            errors.append({
                'field': 'crop_type',
                'code': 'INVALID_CROP_TYPE',
                'message': self.get_translation(language, 'error_invalid_crop')
            })
        
        # Validate nitrogen
        try:
            n = float(form_data['nitrogen'])
            if n < Config.NUTRIENT_MIN or n > Config.NUTRIENT_MAX:
                errors.append({
                    'field': 'nitrogen',
                    'code': 'INVALID_NITROGEN',
                    'message': self.get_translation(language, 'error_nitrogen_range')
                })
        except ValueError:
            errors.append({
                'field': 'nitrogen',
                'code': 'INVALID_NITROGEN',
                'message': self.get_translation(language, 'error_nitrogen_required')
            })
        
        # Validate phosphorous
        try:
            p = float(form_data['phosphorous'])
            if p < Config.NUTRIENT_MIN or p > Config.NUTRIENT_MAX:
                errors.append({
                    'field': 'phosphorous',
                    'code': 'INVALID_PHOSPHOROUS',
                    'message': self.get_translation(language, 'error_phosphorous_range')
                })
        except ValueError:
            errors.append({
                'field': 'phosphorous',
                'code': 'INVALID_PHOSPHOROUS',
                'message': self.get_translation(language, 'error_phosphorous_required')
            })
        
        # Validate potassium
        try:
            k = float(form_data['potassium'])
            if k < Config.NUTRIENT_MIN or k > Config.NUTRIENT_MAX:
                errors.append({
                    'field': 'potassium',
                    'code': 'INVALID_POTASSIUM',
                    'message': self.get_translation(language, 'error_potassium_range')
                })
        except ValueError:
            errors.append({
                'field': 'potassium',
                'code': 'INVALID_POTASSIUM',
                'message': self.get_translation(language, 'error_potassium_required')
            })
        
        if errors:
            return False, errors
        
        return True, []

validation_service = ValidationService()

# ============================================================================
# INITIALIZE OPTIMIZATION SERVICE
# ============================================================================

# Initialize the optimization service
optimization_service = OptimizationService(db)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main page"""
    if 'language' not in session:
        session['language'] = 'en'
    return render_template('index.html')

@app.route('/home.html')
def home():
    """Render home page"""
    return render_template('home.html')

@app.route('/recommendation.html')
def recommendation():
    """Render recommendation page"""
    return render_template('recommendation.html')

@app.route('/weather.html')
def weather_page():
    """Render weather page"""
    return render_template('weather.html')

@app.route('/history.html')
def history_page():
    """Render history page"""
    return render_template('history.html')

@app.route('/analytics.html')
def analytics_page():
    """Render analytics page"""
    return render_template('analytics.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    """Set user's preferred language"""
    try:
        data = request.json
        session['language'] = data.get('language', 'en')
        logger.info(f"Language set to {session['language']}")
        return jsonify({'success': True, 'language': session['language']})
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return jsonify({'error': 'Failed to set language'}), 500

@app.route('/get_language', methods=['GET'])
def get_language():
    """Get user's preferred language"""
    return jsonify({'language': session.get('language', 'en')})

@app.route('/weather', methods=['POST'])
def weather():
    """Get weather data for a location"""
    try:
        data = request.json
        city = data.get('city')
        lat = data.get('lat')
        lon = data.get('lon')
        language = session.get('language', 'en')
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        
        if city:
            result = weather_service.fetch_by_city(city)
        elif lat and lon:
            result = weather_service.fetch_by_coords(lat, lon)
        else:
            return jsonify({
                'success': False,
                'error': t['location_error'],
                'error_code': 'LOCATION_NOT_PROVIDED'
            }), 400
        
        # Check if result is an error response
        if isinstance(result, tuple) and len(result) == 2:
            return jsonify({
                'success': False,
                'error': result[0].get('error', t['weather_error']),
                'error_code': 'WEATHER_API_ERROR'
            }), result[1]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Weather route error: {e}", exc_info=True)
        language = session.get('language', 'en')
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        return jsonify({
            'success': False,
            'error': t['error_weather_service'],
            'error_code': 'WEATHER_SERVICE_ERROR'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make fertilizer prediction with optimization"""
    try:
        # Get language from session
        language = request.form.get('language', session.get('language', 'en'))
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        
        # Validate request with structured error response
        is_valid, errors = validation_service.validate_prediction_request(request.form, language)
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Convert to float
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        
        # Get optional fields with defaults
        location = request.form.get('location', 'Unknown')
        city = request.form.get('city', '')
        country = request.form.get('country', '')
        weather_condition = request.form.get('weather_condition', '')
        
        # Make prediction
        fertilizer_name, confidence = prediction_service.make_prediction(
            temperature, humidity, moisture, soil_type, crop_type, 
            nitrogen, phosphorous, potassium
        )
        
        # Calculate dose
        dose_info = prediction_service.calculate_dynamic_dose(
            crop_type, soil_type, nitrogen, phosphorous, potassium, fertilizer_name
        )
        
        # Generate optimized fertilizer schedule
        optimization_result = optimization_service.generate_fertilizer_schedule(
            crop=crop_type,
            soil_type=soil_type,
            nitrogen=nitrogen,
            phosphorous=phosphorous,
            potassium=potassium,
            predicted_fertilizer=fertilizer_name,
            base_dose=dose_info['base_dose']
        )
        
        # Get fertilizer info with safe fallback
        info = FERTILIZER_INFO.get(fertilizer_name, FERTILIZER_INFO['Urea'])
        comp = FERTILIZER_COMPOSITION.get(fertilizer_name, {'N': 20, 'P': 20, 'K': 20, 'description': 'Balanced fertilizer'})
        
        # Prepare data for database with optimization
        prediction_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'location': location,
            'city': city,
            'country': country,
            'crop': crop_type,
            'soil': soil_type,
            'fertilizer': fertilizer_name,
            'dose': dose_info['dose'],
            'confidence': confidence,
            'nitrogen': nitrogen,
            'phosphorous': phosphorous,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'moisture': moisture,
            'weather_condition': weather_condition,
            'language': language,
            'optimization': optimization_result
        }
        
        # Save to database and get ID
        prediction_id = db.save_prediction(prediction_data)
        
        # Prepare response with optimization data
        response = {
            'success': True,
            'id': prediction_id,
            'fertilizer': fertilizer_name,
            'fertilizer_description': comp.get('description', ''),
            'confidence': confidence,
            'dose_per_acre': dose_info['dose'],
            'n_deficit': dose_info['n_deficit'],
            'p_deficit': dose_info['p_deficit'],
            'k_deficit': dose_info['k_deficit'],
            'n_deficit_amount': dose_info['n_deficit_amount'],
            'p_deficit_amount': dose_info['p_deficit_amount'],
            'k_deficit_amount': dose_info['k_deficit_amount'],
            'fulfilled_pct': dose_info['fulfilled_pct'],
            'explanation': dose_info['explanation'],
            'safety': info['safety'],
            'irrigation': info['irrigation'],
            'timing': info['timing'],
            'method': info.get('method', 'Broadcast evenly and incorporate into soil.'),
            'precautions': info.get('precautions', 'Follow standard safety precautions.'),
            'storage': info.get('storage', 'Store in cool dry place.'),
            'base_dose': dose_info['base_dose'],
            'soil_factor': dose_info['soil_factor'],
            'n_content': dose_info['n_content'],
            'p_content': dose_info['p_content'],
            'k_content': dose_info['k_content'],
            'n_required': CROP_REQUIREMENTS.get(crop_type, {}).get('N', 0),
            'p_required': CROP_REQUIREMENTS.get(crop_type, {}).get('P', 0),
            'k_required': CROP_REQUIREMENTS.get(crop_type, {}).get('K', 0),
            
            # New optimization data
            'optimization': {
                'stages': optimization_result['stages'],
                'total_nutrient_supply': optimization_result['total_nutrient_supply'],
                'optimization_score': optimization_result['optimization_score'],
                'soil_nutrient_balance': optimization_result['soil_nutrient_balance'],
                'irrigation_recommendation': optimization_result['irrigation_recommendation'],
                'summary': optimization_result['summary']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        language = request.form.get('language', session.get('language', 'en'))
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        return jsonify({
            'success': False,
            'error': t['error_prediction_service'],
            'error_code': 'PREDICTION_SERVICE_ERROR'
        }), 500

@app.route('/api/history')
def api_history():
    """Get history data as JSON with pagination and filters"""
    try:
        # Get query parameters
        crop = request.args.get('crop', '')
        date = request.args.get('date', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))  # Default 50 per page, max 200
        limit = min(per_page, 200)  # Cap at 200 per request
        offset = (page - 1) * limit
        
        # Get history with pagination
        result = db.get_history(limit=limit, crop=crop, date=date, offset=offset)
        
        # Calculate pagination info
        total_pages = (result['total'] + limit - 1) // limit if result['total'] > 0 else 1
        
        return jsonify({
            'success': True,
            'data': result['data'],
            'pagination': {
                'page': page,
                'per_page': limit,
                'total': result['total'],
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        })
    except Exception as e:
        logger.error(f"History API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch history'
        }), 500

@app.route('/api/history/stats')
def api_history_stats():
    """Get history statistics"""
    try:
        stats = db.get_history_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"History stats error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch history stats'
        }), 500

@app.route('/api/history/<int:prediction_id>')
def api_history_detail(prediction_id):
    """Get a single prediction by ID"""
    try:
        prediction = db.get_history_by_id(prediction_id)
        if prediction:
            return jsonify({
                'success': True,
                'data': prediction
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction not found'
            }), 404
    except Exception as e:
        logger.error(f"History detail error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch prediction'
        }), 500

@app.route('/api/history/<int:prediction_id>', methods=['DELETE'])
def api_history_delete(prediction_id):
    """Delete a prediction by ID"""
    try:
        success = db.delete_prediction(prediction_id)
        if success:
            return jsonify({
                'success': True,
                'message': 'Prediction deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction not found or could not be deleted'
            }), 404
    except Exception as e:
        logger.error(f"History delete error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to delete prediction'
        }), 500

@app.route('/api/analytics')
def api_analytics():
    """Get comprehensive analytics data as JSON with date range"""
    try:
        days = request.args.get('days', 30, type=int)
        analytics_data = db.get_analytics(days=days)
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    except Exception as e:
        logger.error(f"Analytics API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch analytics'
        }), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    """Download PDF report"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        language = data.get('language', session.get('language', 'en'))
        
        # Create report data with all fields including optimization if available
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'location': data.get('location', 'N/A'),
            'city': data.get('city', ''),
            'country': data.get('country', ''),
            'temperature': data.get('temperature', 0),
            'humidity': data.get('humidity', 0),
            'moisture': data.get('moisture', 0),
            'weather_condition': data.get('weather_condition', ''),
            'soil': data.get('soil', ''),
            'crop': data.get('crop', ''),
            'nitrogen': data.get('nitrogen', 0),
            'phosphorous': data.get('phosphorous', 0),
            'potassium': data.get('potassium', 0),
            'fertilizer': data.get('fertilizer', ''),
            'confidence': data.get('confidence', 0),
            'dose': data.get('dose', 0),
            'n_content': data.get('n_content', 0),
            'p_content': data.get('p_content', 0),
            'k_content': data.get('k_content', 0),
            'base_dose': data.get('base_dose', 0),
            'soil_factor': data.get('soil_factor', 1.0),
            'explanation': data.get('explanation', ''),
            'safety': data.get('safety', ''),
            'irrigation': data.get('irrigation', ''),
            'timing': data.get('timing', ''),
            'method': data.get('method', ''),
            'storage': data.get('storage', ''),
            'n_deficit': data.get('n_deficit', 0),
            'p_deficit': data.get('p_deficit', 0),
            'k_deficit': data.get('k_deficit', 0),
            'n_deficit_amount': data.get('n_deficit_amount', 0),
            'p_deficit_amount': data.get('p_deficit_amount', 0),
            'k_deficit_amount': data.get('k_deficit_amount', 0),
            'fulfilled_pct': data.get('fulfilled_pct', 0),
            'n_required': data.get('n_required', 0),
            'p_required': data.get('p_required', 0),
            'k_required': data.get('k_required', 0),
            'precautions': data.get('precautions', ''),
            
            # Add optimization data if present
            'optimization_stages': json.dumps(data.get('optimization', {}).get('stages', [])) if data.get('optimization') else '[]',
            'optimization_total_nutrient_supply': json.dumps(data.get('optimization', {}).get('total_nutrient_supply', {})) if data.get('optimization') else '{}',
            'optimization_score': data.get('optimization', {}).get('optimization_score', 0) if data.get('optimization') else 0,
            'optimization_soil_nutrient_balance': data.get('optimization', {}).get('soil_nutrient_balance', '') if data.get('optimization') else '',
            'optimization_irrigation_recommendation': json.dumps(data.get('optimization', {}).get('irrigation_recommendation', {})) if data.get('optimization') else '{}',
            'optimization_summary': data.get('optimization', {}).get('summary', '') if data.get('optimization') else ''
        }
        
        # Generate PDF
        pdf_buffer = pdf_service.generate_report(report_data, language)
        
        # Create filename with timestamp
        filename = f"fertilizer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        # Return a simple error message without using translations
        return jsonify({
            'success': False,
            'error': 'Failed to generate PDF report. Please try again.',
            'error_code': 'PDF_GENERATION_ERROR'
        }), 500

@app.route('/download_history_pdf')
def download_history_pdf():
    """Download history as PDF"""
    try:
        language = session.get('language', 'en')
        crop = request.args.get('crop', '')
        date = request.args.get('date', '')
        
        # Get all history data without limit for PDF
        result = db.get_history(limit=None, crop=crop, date=date)
        history_data = result['data']
        
        pdf_buffer = pdf_service.generate_history_pdf(history_data, language)
        
        # Create filename with timestamp
        filename = f"history_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"History PDF generation error: {e}", exc_info=True)
        language = session.get('language', 'en')
        t = TRANSLATIONS.get(language, TRANSLATIONS['en'])
        return jsonify({
            'success': False,
            'error': t.get('error_prediction_service', 'Failed to generate PDF'),
            'error_code': 'PDF_GENERATION_ERROR'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_connected = False
        try:
            with db.get_connection() as conn:
                conn.execute("SELECT 1")
                db_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Get database stats
        stats = db.get_history_stats()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model_service.is_available(),
            'database_connected': db_connected,
            'weather_api_configured': bool(Config.OPENWEATHER_API_KEY),
            'fonts_registered': pdf_service.fonts_registered,
            'optimization_service': 'initialized',
            'total_predictions': stats.get('total_count', 0)
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Smart Agriculture AI Platform - Enterprise Edition")
    logger.info("=" * 60)
    logger.info(f"Database: {Config.DATABASE}")
    logger.info(f"Weather API: {'Configured' if Config.OPENWEATHER_API_KEY else 'Not configured'}")
    logger.info(f"ML Model: {'Loaded' if model_service.is_available() else 'Not found - using rule-based'}")
    logger.info(f"Fonts Registered: {pdf_service.fonts_registered}")
    logger.info(f"Optimization Engine: Initialized")
    logger.info("=" * 60)
    logger.info("Application is running at: http://localhost:5000")
    logger.info("=" * 60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)