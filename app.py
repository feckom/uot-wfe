#!/usr/bin/env python
from flask import Flask, render_template, request, jsonify
import subprocess
import os
import sys
import json
from pathlib import Path

app = Flask(__name__)

# Path to your uot.py script - FIXED THE MISSING PARENTHESIS HERE
UOT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uot.py')

def get_available_languages():
    """Get available languages by parsing uot.py -p output"""
    try:
        result = subprocess.run(
            [sys.executable, UOT_SCRIPT, '-p'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Parse the output to extract language pairs
        languages = set()
        pairs = {}
        
        # Example output line: "en→(es, fr, de)"
        for line in output.split('\n'):
            if '→' in line:
                parts = line.split('→')
                if len(parts) == 2:
                    from_lang = parts[0].strip()
                    to_langs = parts[1].replace('(', '').replace(')', '').split(',')
                    pairs[from_lang] = [lang.strip() for lang in to_langs]
                    languages.add(from_lang)
                    for lang in to_langs:
                        languages.add(lang.strip())
        
        return {
            'all_languages': sorted(languages),
            'language_pairs': pairs
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error getting language list: {e.stderr}", file=sys.stderr)
        return {
            'all_languages': ['en', 'es', 'fr', 'de'],  # fallback
            'language_pairs': {'en': ['es', 'fr', 'de']}
        }

@app.route('/')
def index():
    lang_data = get_available_languages()
    return render_template('index.html', 
                         all_languages=lang_data['all_languages'],
                         language_pairs=json.dumps(lang_data['language_pairs']))

@app.route('/get_target_languages/<source_lang>')
def get_target_languages(source_lang):
    lang_data = get_available_languages()
    return jsonify({
        'target_languages': lang_data['language_pairs'].get(source_lang, [])
    })

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    from_lang = data.get('from', 'en')
    to_lang = data.get('to', 'es')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = subprocess.run(
            [sys.executable, UOT_SCRIPT, '-il', from_lang, '-ol', to_lang, text],
            capture_output=True,
            text=True,
            check=True
        )
        
        return jsonify({
            'translation': result.stdout.strip(),
            'error': None
        })
        
    except subprocess.CalledProcessError as e:
        return jsonify({
            'translation': None,
            'error': f"Translation failed: {e.stderr}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)