#!/usr/bin/env python3
"""
Streamlit UI for Content Sourcing Agent
"""

import streamlit as st
from agant_updated_assement import ContentSourcingAgent, get_config
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Content Sourcing Agent Dashboard", layout="wide")

# Title and timestamp
st.title("Content Sourcing Agent Dashboard")
st.write("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))

# Sidebar for navigation and inputs
st.sidebar.header("Dashboard Controls")
selected_view = st.sidebar.selectbox("Select View", ["Overview", "Assessments", "Student Report", "Teacher Report", "Content Sourcing"])
query = st.sidebar.text_input("Enter Query", value=os.getenv('TEST_QUERY', 'artificial intelligence in automotive systems'))
sources_input = st.sidebar.text_area("Enter Sources (comma-separated URLs)", value=",".join(get_config().STATIC_SOURCES))
sources = [url.strip() for url in sources_input.split(',') if url.strip()]

# Initialize or reuse agent
if 'agent' not in st.session_state:
    config = get_config()
    agent = ContentSourcingAgent(
        config=config,
        api_key=config.GROQ_API_KEY,
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL,
        max_tokens=config.MAX_TOKENS
    )
    st.session_state.agent = agent

agent = st.session_state.agent

# Run agent when query or sources change
if st.sidebar.button("Run Agent"):
    with st.spinner("Processing content and generating assessments..."):
        try:
            assessments = agent.run(query, sources, trigger="manual")  # Changed to store assessments directly
            st.session_state.assessments = assessments  # Store assessments
            st.success("Agent execution completed!")
        except Exception as e:
            st.error(f"Error running agent: {e}")

# Display views based on selection
if 'assessments' not in st.session_state:
    st.session_state.assessments = []

if selected_view == "Overview":
    st.header("Execution Summary")
    st.write(f"**Query:** {query}")
    st.write(f"**Sources Processed:** {len(sources)}")
    if st.session_state.assessments:
        st.write(f"**Assessments Generated:** {len(st.session_state.assessments)}")
    else:
        st.write("**Assessments Generated:** 0 (Run the agent to generate assessments)")

elif selected_view == "Assessments":
    st.header("Generated Assessments")
    if st.session_state.assessments:
        for i, assessment in enumerate(st.session_state.assessments, 1):
            with st.expander(f"Assessment {i}: {assessment.question_text[:50]}..."):
                st.write(f"**Type:** {assessment.question_type}")
                st.write(f"**Question:** {assessment.question_text}")
                if assessment.options:
                    st.write(f"**Options:** {', '.join(assessment.options)}")
                st.write(f"**Correct Answer:** {assessment.correct_answer or 'None'}")
                st.write(f"**Bloom Level:** {assessment.bloom_level}")
                st.write(f"**Objective:** {assessment.objective}")
                st.write(f"**Curriculum Standard:** {assessment.curriculum_standard}")

                # Input for student answer
                student_id = st.text_input(f"Student ID for Assessment {i}", value=os.getenv('STUDENT_ID', 'student_001'))
                teacher_id = st.text_input(f"Teacher ID for Assessment {i}", value=os.getenv('TEACHER_ID', 'teacher_001'))
                sample_answer = assessment.correct_answer or "AI improves ECU diagnostics by analyzing CAN bus data."
                user_answer = st.text_area(f"Enter answer for Assessment {i}", value=sample_answer, height=100)
                if st.button(f"Submit Answer for Assessment {i}", key=f"submit_{i}"):
                    try:
                        response = agent.submit_assessment(i-1, user_answer, student_id, teacher_id)
                        if "error" not in response:
                            st.success(f"Submitted! Score: {response['score']:.2f}, Feedback: {response['feedback']}")
                        else:
                            st.error(response["error"])
                    except Exception as e:
                        st.error(f"Error submitting assessment: {e}")
    else:
        st.write("No assessments generated yet. Run the agent to generate assessments.")

elif selected_view == "Student Report":
    st.header("Student Report")
    student_id = st.text_input("Enter Student ID", value=os.getenv('STUDENT_ID', 'student_001'))
    try:
        student_report = agent.get_student_report(student_id)  # Fixed: Call on agent, not agent.assessment_agent
        if "error" not in student_report:
            st.write(f"**Student ID:** {student_report['student_id']}")
            st.write(f"**Name:** {student_report['name']}")
            st.write(f"**Total Assessments:** {student_report['total_assessments']}")
            st.write(f"**Average Score:** {student_report['average_score']:.2f}")
            st.write(f"**Skill Gaps:** {', '.join(student_report['skill_gaps']) if student_report['skill_gaps'] else 'None'}")
            if student_report['assessments']:
                st.write("**Assessment History:**")
                for a in student_report['assessments']:
                    st.write(f"- Question: {a['question_text']}, Score: {a.get('score', 'N/A')}")
        else:
            st.error(student_report["error"])
    except Exception as e:
        st.error(f"Error generating student report: {e}")

elif selected_view == "Teacher Report":
    st.header("Teacher Report")
    teacher_id = st.text_input("Enter Teacher ID", value=os.getenv('TEACHER_ID', 'teacher_001'))
    try:
        teacher_report = agent.get_teacher_report(teacher_id)  # Fixed: Call on agent, not agent.assessment_agent
        if "error" not in teacher_report:
            st.write(f"**Teacher ID:** {teacher_report['teacher_id']}")
            st.write(f"**Name:** {teacher_report['name']}")
            st.write(f"**Students:** {', '.join(teacher_report['students']) if teacher_report['students'] else 'None'}")
            st.write(f"**Class-wide Gaps:** {', '.join(teacher_report['classwide_gaps']) if teacher_report['classwide_gaps'] else 'None'}")
            st.write(f"**Assessment Summary:** {teacher_report['assessment_summary']}")
        else:
            st.error(teacher_report["error"])
    except Exception as e:
        st.error(f"Error generating teacher report: {e}")

elif selected_view == "Content Sourcing":
    st.header("Content Sourcing Output")
    if st.session_state.get('assessments'):  # Check if agent has run
        st.subheader("Execution Summary")
        st.write(f"**Query:** {query}")
        st.write(f"**Sources Processed:** {len(sources)}")
        st.write(f"**Content Items Fetched:** {len(agent.content_api.storage)}")  # Approximate based on stored content
        st.write(f"**Content Items Processed:** {len(agent.content_api.storage)}")  # Approximate
        st.write(f"**Content Items Stored:** {len(agent.content_api.storage)}")  # Approximate
        st.write(f"**Assessments Generated:** {len(st.session_state.assessments)}")
        st.write(f"**Errors Encountered:** {len(st.session_state.get('errors', [])) if 'errors' in st.session_state else 0}")

        st.subheader("Generated Assessments")
        for i, assessment in enumerate(st.session_state.assessments, 1):
            st.write(f"\n{i}. **Type:** {assessment.question_type}")
            st.write(f"   **Question:** {assessment.question_text}")
            if assessment.options:
                st.write(f"   **Options:** {', '.join(assessment.options)}")
            st.write(f"   **Correct Answer:** {assessment.correct_answer or 'None'}")
            st.write(f"   **Bloom Level:** {assessment.bloom_level}")
            st.write(f"   **Objective:** {assessment.objective}")
            st.write(f"   **Curriculum Standard:** {assessment.curriculum_standard}")

        if st.session_state.get('errors'):
            st.subheader("Errors Encountered")
            for error in st.session_state.errors:
                st.error(f"- {error}")
        else:
            st.success("No errors encountered during content sourcing.")
    else:
        st.write("No content sourcing data available. Run the agent to fetch and process content.")

# Add a footer
st.sidebar.text("Powered by xAI Grok 3")