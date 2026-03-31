import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, date

# ============================================
# PAGE CONFIGURATION
# ============================================
# FEATURE 1: st.set_page_config() - Configure page settings
st.set_page_config(
    page_title="Gemini AI Agent", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# FEATURE 2: st.markdown() with CSS - Custom styling
st.markdown("""
    <style>
        .stApp {
            background-color: #000fff;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("Gemini AI Agent 🤖")
st.caption("Powered by FastAPI and Gemini - Complete Streamlit Feature Showcase")

# ============================================
# SIDEBAR - Navigation and Theme Controls
# ============================================
with st.sidebar:
    st.header("📋 Navigation & Settings")
    
    # FEATURE 3: st.radio() - Single choice selection for page navigation
    page = st.radio("Select Feature", [
        "Chat Interface",
        "Text & Display",
        "Input Widgets",
        "Data Visualization",
        "Layout Components",
        "Advanced Features"
    ])
    
    # FEATURE 4: st.divider() - Visual separator
    st.divider()
    
    st.subheader("🎨 Appearance Settings")
    
    # FEATURE 5: st.checkbox() - Boolean toggle
    show_timestamps = st.checkbox("Show Message Timestamps", value=True)
    show_metrics = st.checkbox("Show Performance Metrics", value=False)
    
    # FEATURE 6: st.color_picker() - Color selection widget
    accent_color = st.color_picker("Choose Accent Color", "#00FF00")
    
    # FEATURE 7: st.selectbox() - Dropdown menu (single select)
    theme = st.selectbox("Select Theme", ["Dark", "Light", "Custom"])
    
    # FEATURE 8: st.slider() - Range slider for numeric input
    font_size = st.slider("Font Size", 10, 20, 14)
    message_limit = st.slider("Max Messages to Show", 5, 50, 20)
    
    st.divider()
    st.info("💡 Navigate using the radio buttons to explore different Streamlit features!")

# ============================================
# PAGE 1: CHAT INTERFACE (Core Feature)
# ============================================
if page == "Chat Interface":
    st.header("💬 AI Chat Interface")
    
    # Backend URL configuration
    BACKEND_URL = "http://127.0.0.1:8000/chat"
    
    # FEATURE 9: st.session_state - Persist data across reruns
    # This is critical for maintaining chat history when page reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # FEATURE 10: st.chat_message() - Display chat messages in proper format
    # Display previous messages from session state
    message_count = min(len(st.session_state.messages), message_limit)
    for message in st.session_state.messages[-message_count:]:
        with st.chat_message(message["role"]):
            # Conditionally show timestamps
            if show_timestamps and "timestamp" in message:
                st.caption(f"⏰ {message['timestamp']}")
            st.markdown(message["content"])
    
    # FEATURE 11: st.chat_input() - Get message from user
    # This widget is specifically designed for chat applications
    if prompt := st.chat_input("Ask about the weather, news, or date..."):
        
        # Generate timestamp for message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if show_timestamps else None
        
        # Display user message immediately
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Set up placeholder for AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... ⏳")
            
            try:
                # Send message to backend API
                response = requests.post(
                    BACKEND_URL, 
                    json={"message": prompt}
                )
                
                # Check if backend responded successfully
                if response.status_code == 200:
                    bot_reply = response.json().get("reply", "No reply received.")
                    message_placeholder.markdown(bot_reply)
                    
                    # Save AI response to session state
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if show_timestamps else None
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_reply,
                        "timestamp": timestamp
                    })
                else:
                    error_msg = f"⚠️ Backend Error: {response.status_code}. Did you hit an API limit?"
                    message_placeholder.markdown(error_msg)
                    
            except requests.exceptions.ConnectionError:
                message_placeholder.markdown("🚨 **Connection Error:** Could not reach the backend. Is your FastAPI server running on port 8000?")
    
    # FEATURE 12: st.button() - Interactive button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        # FEATURE 13: st.metric() - Display KPIs
        st.metric("Messages in Chat", len(st.session_state.messages))

# ============================================
# PAGE 2: TEXT & DISPLAY WIDGETS
# ============================================
elif page == "Text & Display":
    st.header("📝 Text & Display Features")
    
    # FEATURE 14: st.columns() - Create side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Text Display")
        
        # FEATURE 15: st.write() - Versatile multi-purpose display
        st.write("This uses **st.write()** - it supports markdown and multiple types!")
        
        # FEATURE 16: st.text() - Plain text display
        st.text("This is plain text using st.text() - no formatting")
        
        # FEATURE 17: st.markdown() - Markdown formatted text
        st.markdown("""
        ### Markdown Formatting Examples
        - **Bold text** with double asterisks
        - *Italic text* with single asterisks
        - `Code inline` with backticks
        - [Links](https://streamlit.io)
        """)
        
        # FEATURE 18: st.code() - Code blocks with syntax highlighting
        st.code("""
def greet(name):
    return f"Hello, {name}!"

result = greet("Streamlit")
        """, language="python")
        
        # FEATURE 19: st.latex() - Mathematical equations
        st.latex(r'''
            e^{i\pi} + 1 = 0
        ''')
    
    with col2:
        st.subheader("Alert & Status Messages")
        
        # FEATURE 20: st.success() - Success message
        st.success("✅ Operation completed successfully!")
        
        # FEATURE 21: st.error() - Error message
        st.error("❌ An error occurred during processing!")
        
        # FEATURE 22: st.warning() - Warning message
        st.warning("⚠️ Please review your inputs before proceeding!")
        
        # FEATURE 23: st.info() - Informational message
        st.info("ℹ️ This is helpful information for the user!")
        
        # FEATURE 24: st.exception() - Display exceptions
        try:
            int("not a number")
        except ValueError as e:
            st.exception(e)
    
    st.divider()
    
    # FEATURE 25: st.metric() - KPI display with delta
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", "72°F", "+2°F", delta_color="off")
    with col2:
        st.metric("Users Online", "1,234", "-5", delta_color="inverse")
    with col3:
        st.metric("Conversion Rate", "45%", "+3%")

# ============================================
# PAGE 3: INPUT WIDGETS
# ============================================
elif page == "Input Widgets":
    st.header("⚙️ Input Widgets - User Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text & Numeric Inputs")
        
        # FEATURE 26: st.text_input() - Single line text input
        name = st.text_input("Enter your name:", placeholder="John Doe")
        
        # FEATURE 27: st.text_area() - Multi-line text input
        feedback = st.text_area("Enter your feedback:", height=100, placeholder="Share your thoughts...")
        
        # FEATURE 28: st.number_input() - Numeric input control
        age = st.number_input("Enter your age:", min_value=0, max_value=120, value=25, step=1)
        
        # FEATURE 29: st.slider() - Range slider for numeric values
        rating = st.slider("Rate this application:", 1, 5, 3, help="1=Poor, 5=Excellent")
        
        # FEATURE 30: st.date_input() - Date picker widget
        birth_date = st.date_input("Select your birth date:")
        
        # FEATURE 31: st.time_input() - Time picker widget
        meeting_time = st.time_input("Select meeting time:")
    
    with col2:
        st.subheader("Selection Widgets")
        
        # FEATURE 32: st.checkbox() - Boolean checkbox
        newsletter = st.checkbox("Subscribe to newsletter")
        terms = st.checkbox("I agree to terms and conditions")
        
        # FEATURE 33: st.radio() - Single choice radio buttons
        gender = st.radio("Select gender:", ["Male", "Female", "Other", "Prefer not to say"])
        
        # FEATURE 34: st.selectbox() - Dropdown selection (single)
        country = st.selectbox("Select country:", 
                              ["USA", "UK", "Canada", "Australia", "India", "Japan"])
        
        # FEATURE 35: st.multiselect() - Multiple choice selection
        skills = st.multiselect("Select your skills:", 
                               ["Python", "JavaScript", "SQL", "Machine Learning", 
                                "Web Development", "Data Analysis", "Cloud Computing"])
        
        # FEATURE 36: st.color_picker() - Color selection widget
        theme_color = st.color_picker("Pick your theme color:", "#3498db")
        
        # FEATURE 37: st.file_uploader() - File upload widget
        uploaded_file = st.file_uploader("Upload a file:", type=["txt", "csv", "xlsx", "pdf"])
    
    st.divider()
    
    st.subheader("📊 Your Input Summary")
    
    # Create summary data
    summary_data = {
        "Field": ["Name", "Feedback", "Age", "Rating", "Birth Date", "Time", 
                  "Newsletter", "Terms", "Gender", "Country", "Skills", "Theme Color"],
        "Value": [
            name or "Not provided",
            (feedback[:50] + "...") if feedback else "Not provided",
            str(age),
            f"{rating}/5",
            str(birth_date),
            str(meeting_time),
            "Yes" if newsletter else "No",
            "Yes" if terms else "No",
            gender,
            country,
            ", ".join(skills) if skills else "None",
            theme_color
        ]
    }
    
    # FEATURE 38: st.dataframe() - Interactive pandas DataFrame display
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # FEATURE 39: st.info() - Display file upload status
    if uploaded_file:
        st.info(f"✅ File '{uploaded_file.name}' uploaded successfully!")

# ============================================
# PAGE 4: DATA VISUALIZATION
# ============================================
elif page == "Data Visualization":
    st.header("📊 Data Visualization Features")
    
    # Create sample dataset for visualization
    chart_data = pd.DataFrame({
        "Month": ["January", "February", "March", "April", "May", "June"],
        "Sales": [100, 120, 140, 110, 160, 180],
        "Expenses": [80, 90, 85, 95, 100, 110],
        "Profit": [20, 30, 55, 15, 60, 70]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FEATURE 40: st.line_chart() - Line chart visualization
        st.subheader("📈 Line Chart - Sales Over Time")
        st.line_chart(chart_data.set_index("Month")[["Sales"]])
        
        # FEATURE 41: st.area_chart() - Area chart visualization
        st.subheader("📉 Area Chart - Revenue Streams")
        st.area_chart(chart_data.set_index("Month")[["Sales", "Expenses"]])
        
        # FEATURE 42: st.bar_chart() - Bar chart visualization
        st.subheader("📊 Bar Chart - Sales Comparison")
        st.bar_chart(chart_data.set_index("Month")[["Sales", "Profit"]])
    
    with col2:
        # FEATURE 43: st.scatter_chart() - Scatter plot
        st.subheader("🔵 Scatter Chart")
        scatter_data = pd.DataFrame({
            "Expenses": [80, 90, 85, 95, 100, 110],
            "Profit": [20, 30, 55, 15, 60, 70]
        })
        st.scatter_chart(scatter_data)
        
        # FEATURE 44: st.table() - Static table (non-interactive)
        st.subheader("📋 Static Table View")
        st.table(chart_data.head(3))
        
        # FEATURE 45: st.metric() - Key performance indicators
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            total_sales = chart_data["Sales"].sum()
            st.metric("Total Sales", f"${total_sales}")
        with col_m2:
            total_expenses = chart_data["Expenses"].sum()
            st.metric("Total Expenses", f"${total_expenses}")
        with col_m3:
            total_profit = chart_data["Profit"].sum()
            st.metric("Total Profit", f"${total_profit}")
    
    st.divider()
    
    # FEATURE 46: st.dataframe() - Interactive DataFrame with sorting and filtering
    st.subheader("🔍 Interactive DataFrame")
    st.dataframe(chart_data, use_container_width=True, hide_index=True)

# ============================================
# PAGE 5: LAYOUT COMPONENTS
# ============================================
elif page == "Layout Components":
    st.header("🎨 Layout Components - Page Structure")
    
    # FEATURE 47: st.columns() - Create column layout
    st.subheader("Column Layout (1:1:1)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Column 1**")
        st.button("Button in Col 1")
    with col2:
        st.write("**Column 2**")
        st.button("Button in Col 2")
    with col3:
        st.write("**Column 3**")
        st.button("Button in Col 3")
    
    st.divider()
    
    # FEATURE 48: st.columns() with weight parameter
    st.subheader("Weighted Columns (1:2:1)")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write("Small")
    with col2:
        st.write("Large (2x width)")
    with col3:
        st.write("Small")
    
    st.divider()
    
    # FEATURE 49: st.container() - Logical container grouping
    st.subheader("Container (Logical Grouping)")
    with st.container(border=True):
        st.write("Elements inside this container are grouped together")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Sub-element 1")
        with col2:
            st.write("Sub-element 2")
    
    st.divider()
    
    # FEATURE 50: st.expander() - Collapsible sections
    st.subheader("Expander (Collapsible Sections)")
    
    with st.expander("📖 Learn about Streamlit"):
        st.write("""
        Streamlit is an open-source Python library that makes it easy to create custom 
        web apps for machine learning and data science. It allows you to turn data 
        scripts into shareable web apps in minutes without requiring front-end experience.
        """)
        st.code("pip install streamlit", language="bash")
    
    with st.expander("💡 Tips for Using Streamlit"):
        st.markdown("""
        1. Use session_state to persist data across reruns
        2. Leverage caching with @st.cache_data for performance
        3. Use st.columns() for responsive layouts
        4. Sidebar is great for configuration controls
        5. Use st.spinner() for long-running operations
        """)
    
    st.divider()
    
    # FEATURE 51: st.tabs() - Tabbed interface
    st.subheader("Tabs (Multi-view Interface)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Details", "Analytics", "Settings"])
    
    with tab1:
        st.write("**Overview Tab**")
        st.metric("Total Users", "1,234")
    
    with tab2:
        st.write("**Details Tab**")
        st.dataframe(pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [95, 87]}))
    
    with tab3:
        st.write("**Analytics Tab**")
        st.line_chart({"Hours": [1, 2, 3, 4, 5], "Visitors": [10, 20, 30, 25, 35]})
    
    with tab4:
        st.write("**Settings Tab**")
        st.selectbox("Theme", ["Light", "Dark"])

# ============================================
# PAGE 6: ADVANCED FEATURES
# ============================================
elif page == "Advanced Features":
    st.header("🚀 Advanced Features & Interactions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("State Management")
        
        # FEATURE 52: st.session_state - Persist data across reruns
        if 'counter' not in st.session_state:
            st.session_state.counter = 0
        
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("➕ Increment"):
                st.session_state.counter += 1
        with col_btn2:
            if st.button("➖ Decrement"):
                st.session_state.counter -= 1
        with col_btn3:
            if st.button("🔄 Reset"):
                st.session_state.counter = 0
        
        st.write(f"Current Counter Value: **{st.session_state.counter}**")
    
    with col2:
        st.subheader("Progress & Loading")
        
        # FEATURE 53: st.progress() - Progress bar display
        progress_value = st.slider("Progress:", 0, 100, 50)
        st.progress(progress_value / 100, text=f"{progress_value}%")
        
        # FEATURE 54: st.button() with spinner demo
        if st.button("Simulate Task"):
            # FEATURE 55: st.spinner() - Loading spinner while processing
            with st.spinner("Processing... This takes a moment"):
                time.sleep(2)
            st.success("Task completed!")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Animations & Celebrations")
        
        # FEATURE 56: st.balloons() - Celebration animation
        if st.button("🎉 Launch Balloons!"):
            st.balloons()
            st.success("Celebration mode activated!")
    
    with col2:
        st.subheader("Special Effects")
        
        # FEATURE 57: st.snow() - Snow animation
        if st.button("❄️ Let it Snow!"):
            st.snow()
            st.info("Winter vibes!")
    
    st.divider()
    
    st.subheader("📚 Complete Feature Reference")
    
    # FEATURE 58: st.markdown() with complex formatting
    features_doc = """
    ## All Streamlit Features Demonstrated
    
    ### Display Functions
    | Feature | Purpose |
    |---------|---------|
    | `st.write()` | Multi-purpose display (text, charts, data) |
    | `st.text()` | Plain text without formatting |
    | `st.markdown()` | Markdown formatted text |
    | `st.code()` | Code blocks with syntax highlighting |
    | `st.latex()` | Mathematical equations |
    | `st.metric()` | KPI display with delta values |
    
    ### Input Functions
    | Feature | Purpose |
    |---------|---------|
    | `st.text_input()` | Single-line text input |
    | `st.text_area()` | Multi-line text input |
    | `st.number_input()` | Numeric input |
    | `st.slider()` | Range slider |
    | `st.date_input()` | Date picker |
    | `st.time_input()` | Time picker |
    | `st.checkbox()` | Boolean toggle |
    | `st.radio()` | Single choice selection |
    | `st.selectbox()` | Dropdown menu |
    | `st.multiselect()` | Multiple choice selection |
    | `st.color_picker()` | Color selection |
    | `st.file_uploader()` | File upload |
    | `st.chat_input()` | Chat message input |
    
    ### Visualization Functions
    | Feature | Purpose |
    |---------|---------|
    | `st.line_chart()` | Line chart |
    | `st.bar_chart()` | Bar chart |
    | `st.area_chart()` | Area chart |
    | `st.scatter_chart()` | Scatter plot |
    | `st.dataframe()` | Interactive table |
    | `st.table()` | Static table |
    
    ### Layout Functions
    | Feature | Purpose |
    |---------|---------|
    | `st.columns()` | Side-by-side layout |
    | `st.container()` | Logical grouping |
    | `st.expander()` | Collapsible section |
    | `st.tabs()` | Tabbed interface |
    | `st.divider()` | Visual separator |
    | `st.sidebar` | Sidebar panel |
    
    ### Advanced Features
    | Feature | Purpose |
    |---------|---------|
    | `st.session_state` | Persist data across reruns |
    | `st.spinner()` | Loading animation |
    | `st.progress()` | Progress bar |
    | `st.balloons()` | Celebration animation |
    | `st.snow()` | Snow animation |
    | `st.chat_message()` | Chat message display |
    """
    
    st.markdown(features_doc)
    
    st.divider()
    
    # FEATURE 59: Final summary with multiple st.write() calls
    st.subheader("✅ Learning Achievements")
    st.write(
        "You now understand:",
        "✓ Display widgets (text, markdown, code)",
        "✓ Alert messages (success, error, warning, info)",
        "✓ Input controls (text, numbers, selections)",
        "✓ Data visualization (charts, tables, metrics)",
        "✓ Layout management (columns, containers, tabs)",
        "✓ State persistence (session_state)",
        "✓ Loading indicators (spinner, progress)",
        "✓ Interactive features (buttons, forms)",
        sep="\n"
    )
    
    st.success("🎓 You are now ready to build advanced Streamlit applications!")
