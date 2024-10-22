from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from chatbot import create_chatbot, get_response

app = Flask(__name__)
api = Api(app)

chatbot = create_chatbot()

class ChatbotAPI(Resource):
    def post(self):
        data = request.get_json()
        query = data.get('query')  # Update to fetch the correct key from the JSON

        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        response = get_response(chatbot, query)
        return jsonify({"response": response})

api.add_resource(ChatbotAPI, '/chat')

if __name__ == '__main__':
    app.run(debug=True)
