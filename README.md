# 🤖 Multi-Agent Business Analytics System

**Name :** J.Shanil Praveen Diwakar  

**University :** University of Moratuwa 

**Year :** 3

---

## 🎯 Summary of the Approach

This project was developed as a sophisticated multi-agent system, architected to solve the complex challenges of automating business analytics and customer engagement. The solution is built on the robust frameworks of LangChain and LangGraph, where LangChain provides the specialized tools and reasoning capabilities, and LangGraph serves as a state-based orchestration layer. This modular design ensures a reliable, scalable, and highly maintainable application.

The system is centered around three specialized agents, each with a distinct purpose:

🗣️ **A Feedback Agent** was engineered to generate empathetic and context-aware responses to customer reviews. This allows for instant, professional engagement at scale.

📈 **A Sentiment Agent** was designed to act as an on-demand data analyst. It intelligently interprets user intent to select the most appropriate visualization tool, capable of generating different types of graphs—such as line charts for sentiment trends over time or pie charts for a breakdown of sentiment distribution.

📊 **A Strategic Agent** was implemented to streamline the business reporting process. It is responsible for autonomously generating comprehensive strategic reports and automatically saving each one as a PDF document, ensuring a consistent and immediate delivery of key insights.

By orchestrating these agents into a single, cohesive workflow, this approach represents a significant advancement over traditional single-agent systems. The separation of concerns between agents, managed by LangGraph's deterministic control flow, guarantees the reliability and accuracy of every task.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ShanilPraveen/steamnoodles-feedback-agent-Shanil-Praveen.git
```

### 2. Navigate to the Project Directory
```bash
cd steamnoodles-feedback-agent-Shanil-Praveen
```

### 3. Create and Activate a Virtual Environment
```bash
python -m venv venv
```
**Activate the environment:**
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Key
Create a `.env` file in the project's root directory and add your Groq API key:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

---

## 🧪 How to Test Each Agent

The agents can be tested by running the test.py file from the terminal. The file contains a check_workflow function that sends a given prompt to the LangGraph application, and a series of commented-out example calls.

• Open the test.py file in a code editor.

• Uncomment the test prompts you want to run. By default, most of the lines are commented out with #. To run a specific test, simply remove the # at the beginning of the line.

• For example, to test the sentiment agent with your specific query, make sure this line is active:
```python
check_workflow("how is the sentiments of customers are changed in 2023?")
```

• Run the file from your terminal. Make sure your virtual environment is activated before running the command.
```bash
python test.py
```

### 1. Feedback Agent 🗣️
**Purpose:** To generate polite and context-aware responses to customer reviews.

**Sample Prompts:**
```python
check_workflow("Respond to this review: The service was fantastic and the food was delicious!")
check_workflow("Respond to this review: The food was cold and the waiter was very rude.")
```

### 2. Sentiment Agent 📈
**Purpose:** To analyze sentiment data and produce visualizations. It intelligently chooses between different chart types based on the query.

**Sample Prompts:**
```python
check_workflow("Show me the sentiment trend for the year 2019.")
# Expected Output: Line chart image file path

check_workflow("What is the sentiment distribution for the third quarter of 2018?")
# Expected Output: Pie chart image file path
```

### 3. Strategic Agent 📊
**Purpose:** To generate comprehensive business reports with recommendations based on a specified date range.

**Sample Prompts:**
```python
check_workflow("Generate a strategic recommendations report for the period from 2019-01-01 to 2019-06-30.")
# Expected Output: The full report content printed to the console

check_workflow("Create a report for the last half of 2018 and save it as well.")
# Expected Output: A confirmation message with the path to the saved PDF file

check_workflow("Generate a strategic recommendations report for the period from 2019-01-01 to 2019-06-30 and save it as a PDF.")
# Expected Output: A confirmation message with the path to the saved PDF file
```
