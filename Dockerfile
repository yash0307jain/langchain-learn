FROM python:3.12

WORKDIR /app

COPY requirements.txt /app
COPY .env /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY weekly_task_planner_agent.py /app

CMD ["streamlit", "run", "weekly_task_planner_agent.py"]