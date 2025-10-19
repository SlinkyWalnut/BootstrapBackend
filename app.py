from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os 
import json 
# remove ts
from typing import List, Dict, Any
from dotenv import load_dotenv
   
   # Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Analyze only the newest Speaker 0 text while using full conversation as context
def analyze_conversation_response(emotions, transcriptions, eval_metric, new_text=None):
    try:
        print(f"🔑 OpenAI API Key exists: {bool(os.getenv('OPENAI_API_KEY'))}")
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # If no new text to analyze, return neutral multiplier
        if not new_text or not new_text.strip():
            return {"score_multiplier": 1.0, "moves": []}
        
        print(f"📝 New text to analyze: '{new_text}'")
        
        # Build context with full conversation for reference
        conversation_context = ""
        if transcriptions:
            conversation_context = f"\n\nFull conversation context (for reference only, do not analyze):\n"
            conversation_context += "\n".join([f"Speaker {turn['speaker']}: {turn['text']}" for turn in transcriptions])
        
        context = f"""You are analyzing Speaker 0's conversation performance in chess move terms for: {eval_metric}

Speaker 1's emotions during the conversation: {emotions}

Analyze ONLY the new text provided below using these chess move types:
- BRILLIANT: Exceptional, perfect response that the other speaker responds well to. 
    - Example: Date conversation: "You are so beautiful, your smile is so contagious"
- GREAT: Very good response, well-timed and effective
- BEST: Optimal choice in the situation
- EXCELLENT: Strong response, clearly beneficial
- GOOD: Solid choice, generally positive
- BOOK: Standard, extremely predictable expected responses (neutral)
    - Example: "Hey how are you?", "I'm doing well, thanks for asking"
- INACCURACY: Minor mistake, slightly off but not harmful
- MISTAKE: Clear error, poor choice with negative impact
    - Example: Interview conversation: "I didn’t have time to look at your website."
- BLUNDER: Major error, very damaging to the conversation
    - Example: Date conversation: "You look really different from your pictures."
- MISSED_WIN: Failed to capitalize on a great opportunity
    - Example: Date conversation: "Speaker 1: Do you want to go to the movies?" Speaker 0: "I'm not sure, I have a lot of work to do."

EXAMPLES:

Date

- You look way different from your pictures. (Negative)

- My ex used to love this restaurant. (Negative)

- You have a really genuine laugh. (Positive)

- I really enjoy talking to you. (Positive)

Interview

- I didn’t have time to look at your website. (Negative)

- I’m excited about this role — it aligns with what I’ve been building toward. (Positive)

Team Meeting

- That idea doesn’t make sense. (Negative)

- That’s an interesting approach — maybe we can build on that. (Positive)

Friend Chat

- You’re overreacting. (Negative)

- That sounds really tough — want to talk about it? (Positive)



For the new text, return:
1. The exact substring that represents the key part of the move
2. The move type classification
3. Brief reason (1-2 words max)

IMPORTANT: 
- Only analyze the NEW TEXT provided below, not the conversation context
- Return the exact substring that should be highlighted
- You can identify multiple moves within the same text if appropriate
- Provide a score multiplier from 0.5 to 2.0 based on the move quality:
  * 0.5-0.7: Very negative moves (BLUNDER, major MISTAKE)
  * 0.8-0.9: Slightly negative moves (minor MISTAKE, INACCURACY)
  * 1.0: Neutral moves (BOOK, standard responses)
  * 1.1-1.3: Positive moves (GOOD, EXCELLENT)
  * 1.4-2.0: Exceptional moves (GREAT, BEST, BRILLIANT)

Format your response as JSON only, no other text:
{{
    "moves": [
        {{
            "text": "exact substring to highlight",
            "move_type": "MOVE_TYPE", 
            "reason": "brief reason"
        }}
    ],
    "score_multiplier": 1.0
}}{conversation_context}"""
        
        response = client.chat.completions.create(
            model = "gpt-5-nano",
            messages = [
                {"role":"system", "content":context},
                {"role":"user", "content": f"Analyze ONLY this new text from Speaker 0:\n\n{new_text}"}
            ],
        )
        
        # Parse JSON response
        import json
        response_text = response.choices[0].message.content
        print(f"🤖 GPT Response: {response_text}")
        
        # Extract JSON from response (in case there's extra text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        else:
            return {"score_multiplier": 1.0, "moves": []}
            
    except Exception as e:
        print(f"❌ Error in analyze_conversation_response: {str(e)}")
        return {"score_multiplier": 1.0, "moves": []}
        
@app.route('/analyze_conversation', methods=['POST'])
def analyze_conversation():
    try:
        print("📥 Received request to /analyze_conversation")
        data = request.get_json()
        print(f"📋 Request data: {data}")
        
        emotions = data.get('emotions', [])
        transcriptions = data.get('transcriptions', [])
        eval_metric = data.get('eval_metric', 'general conversation')
        new_text = data.get('new_text', '')
        
        print(f"🎭 Emotions: {emotions}")
        print(f"💬 Transcriptions: {len(transcriptions)} total")
        print(f"📊 Eval metric: {eval_metric}")
        print(f"📝 New text: '{new_text}'")

        response_data = analyze_conversation_response(emotions, transcriptions, eval_metric, new_text)
        print(f"🤖 GPT Response: {response_data}")
        
        score_multiplier = response_data.get('score_multiplier', 1.0)
        moves = response_data.get('moves', [])
        
        print(f"✅ Returning response - Score Multiplier: {score_multiplier}, Moves: {len(moves)}")
        return jsonify({
            'success': True, 
            'score_multiplier': score_multiplier,
            'moves': moves,
            'eval_metric': eval_metric
        })
    except Exception as e:
        print(f"❌ Error in analyze_conversation endpoint: {str(e)}")
        print(f"❌ Error type: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Progress bar value (score) is now included in the analyze_conversation endpoint



if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    






