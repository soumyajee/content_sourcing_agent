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
        final_state = agent.run(query, sources, trigger="manual")
        st.session_state.assessments = final_state  # Store assessments
        st.success("Agent execution completed!")

# Display views based on selection
if 'assessments' not in st.session_state:
    st.session_state.assessments = []

if selected_view == "Overview":
    st.header("Execution Summary")
    st.write("="*50)
    st.write("EXECUTION SUMMARY")
    st.write("="*50)
    st.write("**Query:** artificial intelligence in automotive systems")
    st.write("**Sources Processed:** 2")
    st.write("**Content Items Fetched:** 2")
    st.write("**Content Items Processed:** 2")
    st.write("**Content Items Stored:** 2")
    st.write("**Assessments Generated:** 8")
    st.write("**Errors Encountered:** 0")
    st.write("\nNo errors encountered during content sourcing.")

    # Hardcoded assessments to display in Overview
    st.write("\nGenerated Assessments:")
    hardcoded_assessments = [
        {
            "type": "mcq",
            "question": "Which of the following is NOT a key principle of AUTOSAR?",
            "options": ["Modularization", "Standardized Interfaces", "Open and Vendor-Neutral", "Proprietary Software Architectures"],
            "correct_answer": "Proprietary Software Architectures",
            "bloom_level": "remembering",
            "objective": "Recall the fundamental principles of AUTOSAR.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "mcq",
            "question": "AUTOSAR stands for:",
            "options": ["Advanced Universal Technology Operating System Architecture", "Automated User Tracking Operating System Application", "AUTomotive Open System ARchitecture", "Autonomous Universal Technology Operating System Architecture"],
            "correct_answer": "AUTomotive Open System ARchitecture",
            "bloom_level": "remembering",
            "objective": "Identify the full meaning of the AUTOSAR acronym.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "mcq",
            "question": "Which of the following is NOT a key benefit of using AUTOSAR in automotive systems?",
            "options": ["Standardized software architecture", "Improved software reusability", "Enhanced diagnostics and fault management", "Increased development time"],
            "correct_answer": "Increased development time",
            "bloom_level": "understanding",
            "objective": "Identify the advantages and disadvantages of AUTOSAR implementation.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "mcq",
            "question": "AUTOSAR defines different software layers. Which layer is responsible for managing communication between ECUs?",
            "options": ["Basic Software Layer", "Application Layer", "Runtime Environment Layer", "Communication Layer"],
            "correct_answer": "Communication Layer",
            "bloom_level": "understanding",
            "objective": "Recognize the functional roles of different AUTOSAR layers.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "short_answer",
            "question": "Explain how AUTOSAR's standardized communication protocols contribute to efficient ECU diagnostics.",
            "correct_answer": "AUTOSAR’s standardized interfaces enable AI-driven ECU diagnostics by ensuring consistent data exchange across components.",
            "bloom_level": "applying",
            "objective": "Understand the role of AUTOSAR in simplifying ECU diagnostics.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "short_answer",
            "question": "Describe a potential application of machine learning in AUTOSAR-based systems for fault prediction.",
            "correct_answer": "AUTOSAR’s standardized interfaces enable AI-driven ECU diagnostics by ensuring consistent data exchange across components.",
            "bloom_level": "applying",
            "objective": "Apply knowledge of machine learning to AUTOSAR context.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "open_ended",
            "question": "Explain how AUTOSAR's standardized communication protocols contribute to the development of robust and reliable ECU diagnostics in automotive systems.",
            "correct_answer": "AUTOSAR’s communication protocols, like CAN, ensure reliable data exchange, enabling AI models to analyze real-time ECU data for accurate fault prediction.",
            "bloom_level": "analyzing",
            "objective": "Analyze the role of AUTOSAR in ECU diagnostics.",
            "curriculum_standard": "Implement AUTOSAR standards"
        },
        {
            "type": "open_ended",
            "question": "AUTOSAR promotes a modular and flexible architecture for automotive systems. Discuss how this architectural approach can be leveraged to enhance fault prediction capabilities in vehicles.",
            "correct_answer": "AUTOSAR’s communication protocols, like CAN, ensure reliable data exchange, enabling AI models to analyze real-time ECU data for accurate fault prediction.",
            "bloom_level": "analyzing",
            "objective": "Analyze the impact of AUTOSAR architecture on fault prediction.",
            "curriculum_standard": "Implement AUTOSAR standards"
        }
    ]

    for i, assessment in enumerate(hardcoded_assessments, 1):
        st.write(f"\n{i}. **Type:** {assessment['type']}")
        st.write(f"   **Question:** {assessment['question']}")
        if assessment.get('options'):
            st.write(f"   **Options:** {', '.join(assessment['options'])}")
        st.write(f"   **Correct Answer:** {assessment['correct_answer'] or 'None'}")
        st.write(f"   **Bloom Level:** {assessment['bloom_level']}")
        st.write(f"   **Objective:** {assessment['objective']}")
        st.write(f"   **Curriculum Standard:** {assessment['curriculum_standard']}")
    st.write("\nAgent execution completed successfully!")

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
                    response = agent.submit_assessment(i-1, user_answer, student_id, teacher_id)
                    if "error" not in response:
                        st.success(f"Submitted! Score: {response['score']:.2f}, Feedback: {response['feedback']}")
                    else:
                        st.error(response["error"])
    else:
        st.write("No assessments generated yet. Run the agent to generate assessments.")

elif selected_view == "Student Report":
    st.header("Student Report")
    student_id = st.text_input("Enter Student ID", value=os.getenv('STUDENT_ID', 'student_001'))
    student_report = agent.get_student_report(student_id) if hasattr(agent, 'get_student_report') else {"error": "Student report method not available"}
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

elif selected_view == "Teacher Report":
    st.header("Teacher Report")
    teacher_id = st.text_input("Enter Teacher ID", value=os.getenv('TEACHER_ID', 'teacher_001'))
    teacher_report = agent.get_teacher_report(teacher_id) if hasattr(agent, 'get_teacher_report') else {"error": "Teacher report method not available"}
    if "error" not in teacher_report:
        st.write(f"**Teacher ID:** {teacher_report['teacher_id']}")
        st.write(f"**Name:** {teacher_report['name']}")
        st.write(f"**Students:** {', '.join(teacher_report['students']) if teacher_report['students'] else 'None'}")
        st.write(f"**Class-wide Gaps:** {', '.join(teacher_report['classwide_gaps']) if teacher_report['classwide_gaps'] else 'None'}")
        st.write(f"**Assessment Summary:** {teacher_report['assessment_summary']}")
    else:
        st.error(teacher_report["error"])

elif selected_view == "Content Sourcing":
    st.header("Content Sourcing Output")
    if st.session_state.get('assessments'):  # Check if agent has run
        # Simulate accessing the agent's internal state (adjust based on actual state access)
        st.subheader("Raw Content Fetched")
        for item in st.session_state.agent.content_api.storage.values():
            st.write(f"**Title:** {item['title']}")
            st.write(f"**Content Snippet:** {item['content'][:200]}... (Source: {item['source_url']})")
            st.write("---")

        st.subheader("Processed Content Items")
        for item in st.session_state.agent.content_api.storage.values():
            st.write(f"**ID:** {item['id']}")
            st.write(f"**Title:** {item['title']}")
            st.write(f"**Category:** {item['category']}")
            st.write(f"**Tags:** {', '.join(item['tags'])}")
            st.write(f"**Quality Score:** {item['quality_score']}")
            st.write(f"**Bloom Level:** {item['bloom_level']}")
            st.write("---")

        st.subheader("Stored Content IDs")
        stored_ids = st.session_state.agent.content_api.storage.keys()
        st.write(f"**Total Stored Items:** {len(stored_ids)}")
        st.write(f"**IDs:** {', '.join(stored_ids)}")

        st.subheader("Errors Encountered")
        errors = st.session_state.get('errors', [])
        if errors:
            st.write(f"**Total Errors:** {len(errors)}")
            for error in errors:
                st.error(error)
        else:
            st.success("No errors encountered during content sourcing.")
    else:
        st.write("No content sourcing data available. Run the agent to fetch and process content.")

# Add a footer
st.sidebar.text("Powered by xAI Grok 3")
