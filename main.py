import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


import random

# Predefined feedback templates with a placeholder for the event name
feedback_templates = [
    "I absolutely loved the {event}! It was truly an unforgettable experience.",
    "The {event} was amazing! The organization and performances exceeded my expectations.",
    "I had a great time at the {event}. Everything from the setup to the execution was flawless.",
    "The {event} really impressed me. The energy and creativity were off the charts!",
    "I enjoyed the {event} immensely. It was well organized and had a vibrant atmosphere.",
    "The {event} was a fantastic experience, though I think some improvements could make it even better.",
    "I was thoroughly entertained by the {event}. It showcased exceptional talent and passion.",
    "The {event} exceeded my expectations with its innovative approach and dynamic performances.",
    "Overall, the {event} was a delightful experience that I would highly recommend to others.",
    "While the {event} was enjoyable, I believe there are opportunities for further enhancements."
]

def generate_random_feedback(event_name):
    """
    Generate a random feedback string for the given event.

    Args:
        event_name (str): The name of the event.

    Returns:
        str: A random feedback string customized for the event.
    """
    template = random.choice(feedback_templates)
    return template.format(event=event_name)

# Example usage
if __name__ == "__main__":
    # List of sample events
    events = ["Dance Battle", "Singing Competition", "Drama Play", "Art Exhibition", "Coding Contest"]

    # Generate and print random feedback for each event
    for event in events:
        feedback = generate_random_feedback(event)
        print(f"Feedback for {event}: {feedback}")

# Generate random participant names
def generate_name():
    first_names = ["Aarav", "Ananya", "Rohan", "Meera", "Vikram", "Sanya", "Kabir", "Ishita", "Dev", "Neha"]
    last_names = ["Sharma", "Iyer", "Verma", "Nair", "Patel", "Das", "Kapoor", "Bose", "Malhotra", "Singh"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# List of cultural events
events = [
    "Dance Battle", "Singing Competition", "Drama Play", "Art Exhibition",
    "Coding Contest", "Photography", "Poetry Slam", "Fashion Show",
    "Music Band Show", "Quiz Competition"
]

# Colleges participating
colleges = [
    "ABC University", "XYZ College", "Tech Institute", "Fine Arts Academy",
    "National College", "Creative Hub", "Elite University", "Future Innovators"
]

# Feedback pool
feedback_texts = [
    "Amazing experience! Would love to participate again.",
    "Great event, but the organization could be better.",
    "Loved the energy and performances!",
    "Judging was fair and well-organized.",
    "I enjoyed meeting new people from different colleges.",
    "The event was exciting but needs better scheduling.",
    "Fantastic crowd and great vibes!",
    "More interactive sessions would make it even better.",
    "The competition was tough, but I learned a lot.",
    "Overall, a well-organized and memorable experience!"
]

# Generate dataset
data = []
for i in range(250):  # 250 participants
    participant = {
        "Participant ID": f"P-{i+1}",
        "Name": generate_name(),
        "Age": random.randint(18, 26),
        "Gender": random.choice(["Male", "Female", "Other"]),
        "Event Attended": random.choice(events),
        "Day of Event": random.choice(["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]),
        "College Name": random.choice(colleges),
        "Feedback": random.choice(feedback_texts),
        "Rating (out of 5)": random.randint(1, 5),
        "Participation Type": random.choice(["Solo", "Group"])
    }
    data.append(participant)

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("inbloom_25_participants.csv", index=False)

# Display first few rows
print(df.head())

# Set Streamlit page configuration
st.set_page_config(page_title="INBLOOM '25", layout="wide")

# Sidebar Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Fix applied
st.sidebar.title("🌟 INBLOOM '25 Navigation")
page = st.sidebar.radio("📌 Select a Section", ["🎭 Event Details", "📊 Participation Analysis", "📸 Gallery","reviews"])

# 📌 Event Details Page (Enhanced UI with Better Text Visibility)
if page == "🎭 Event Details":
    st.title("🎭 INBLOOM ‘25 - Event Details")
    st.markdown("### 🌟 Explore the exciting cultural events happening over 5 days!")

    # Event descriptions
    event_descriptions = {
        "Dance Battle": "💃🔥 A high-energy face-off where the best dancers showcase their moves in a thrilling competition!",
        "Singing Competition": "🎤 A stage where melodious voices compete to win hearts and the grand prize!",
        "Drama Play": "🎭 Experience the magic of theatre with powerful storytelling, expressive acting, and gripping plots.",
        "Art Exhibition": "🎨 A gallery of breathtaking artworks from talented artists across different colleges.",
        "Coding Contest": "👨‍💻 Solve real-world problems with logic, speed, and coding mastery in a highly competitive environment!",
        "Photography": "📸 A visual storytelling challenge capturing the essence of culture, nature, and creativity.",
        "Poetry Slam": "📝 A battle of words where poets pour their hearts out with mesmerizing verses and performances.",
        "Fashion Show": "👗 Walk the ramp in stunning designs and showcase the latest trends in this glamorous event!",
        "Music Band Show": "🎸 Electrifying performances by college bands bringing rock, pop, and classical fusion to life!",
        "Quiz Competition": "🧠 Test your knowledge in a fast-paced quiz covering diverse topics with intense competition!"
    }

    # Event Dates & Venues
    event_dates = ["April 5", "April 6", "April 7", "April 8", "April 9"]
    venues = ["Auditorium", "Main Stage", "Theater Hall", "Art Room", "Tech Hub"]

    # Display events in a card-like format
    for i, (event, description) in enumerate(event_descriptions.items()):
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    border-left: 5px solid #ff4b4b;
                    color: #000; /* Set text color to black */
                ">
                    <h4 style="color:#ff4b4b;">🎤 {event}</h4>
                    <p style="color:#000;"><strong>📅 Date:</strong> {event_dates[i % 5]}</p>
                    <p style="color:#000;"><strong>📍 Venue:</strong> {venues[i % 5]}</p>
                    <p style="color:#000;"><strong>📝 Description:</strong> {description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.success("✨ Stay tuned for more events and surprises!")

elif page == "reviews":
    st.title("📝 Participant Feedback Analysis")
    st.write("Analyze participant feedback using text analysis techniques.")

    # Check if the required columns exist; if not, generate a dummy dataset.
    if 'Event' not in df.columns or 'Feedback' not in df.columns:
        st.warning("Feedback data not found in the dataset. Generating dummy feedback data...")
        import random

        # Predefined events and feedback templates
        events_list = [
            "Dance Battle", "Singing Competition", "Drama Play", "Art Exhibition", "Coding Contest",
            "Photography", "Poetry Slam", "Fashion Show", "Music Band Show", "Quiz Competition"
        ]
        feedback_templates = [
            "I absolutely loved the {event}! It was truly an unforgettable experience.",
            "The {event} was amazing! The organization and performances exceeded my expectations.",
            "I had a great time at the {event}. Everything from the setup to the execution was flawless.",
            "The {event} really impressed me. The energy and creativity were off the charts!",
            "I enjoyed the {event} immensely. It was well organized and had a vibrant atmosphere.",
            "The {event} was a fantastic experience, though I think some improvements could make it even better.",
            "I was thoroughly entertained by the {event}. It showcased exceptional talent and passion.",
            "The {event} exceeded my expectations with its innovative approach and dynamic performances.",
            "Overall, the {event} was a delightful experience that I would highly recommend to others.",
            "While the {event} was enjoyable, I believe there are opportunities for further enhancements."
        ]
        def generate_random_feedback(event_name):
            template = random.choice(feedback_templates)
            return template.format(event=event_name)
        
        # Generate dummy dataset with 250 rows
        dummy_data = []
        for i in range(250):
            event = random.choice(events_list)
            dummy_data.append({
                "Event": event,
                "Feedback": generate_random_feedback(event)
            })
        df = pd.DataFrame(dummy_data)

    # Display a sample of the feedback data
    st.markdown("#### Sample Participant Feedback")
    st.dataframe(df[['Event', 'Feedback']].head(10))

    # ---------------------------
    # Task 1: Generate Word Cloud for Selected Event
    # ---------------------------
    st.markdown("### Word Cloud by Event")
    selected_event = st.selectbox("Select an event for word cloud", options=sorted(df["Event"].unique()))
    event_feedback = df[df["Event"] == selected_event]["Feedback"]
    feedback_text = " ".join(event_feedback)
    
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(feedback_text)
    
    st.subheader("Word Cloud for " + selected_event)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # ---------------------------
    # Task 2: Compare Feedback Within Each Event
    # ---------------------------
    st.markdown("### Common Words in Feedback per Event")
    st.write("Below are the five most common words (excluding stopwords) in the feedback for each event:")

    import nltk
    from collections import Counter
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    # Build a dictionary with the common words for each event
    common_words_dict = {}
    for event in sorted(df["Event"].unique()):
        feedback_texts = df[df["Event"] == event]["Feedback"].tolist()
        words = []
        for feedback in feedback_texts:
            tokens = nltk.word_tokenize(feedback.lower())
            # Filter to keep only alphabetic words that are not stopwords
            filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
            words.extend(filtered_tokens)
        common_counts = Counter(words).most_common(5)
        common_words_dict[event] = common_counts

    # Display the common words per event in a formatted way
    for event, common in common_words_dict.items():
        st.markdown(f"**{event}:**")
        st.write(common)

    st.success("Text analysis complete!")


# 📊 Participation Analysis Page
elif page == "📊 Participation Analysis":
    st.title("📊 Participation Analysis")
    st.write("📈 Visual insights into participant engagement across events for INBLOOM ‘25.")

    # ---------------------------
    # Generate Dummy Participation Data
    # ---------------------------
    import random
    events = [
        "Dance Battle", "Singing Competition", "Drama Play", "Art Exhibition", "Coding Contest",
        "Photography", "Poetry Slam", "Fashion Show", "Music Band Show", "Quiz Competition"
    ]
    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
    colleges = [
        "ABC University", "XYZ College", "Tech Institute", "Fine Arts Academy",
        "National College", "Creative Hub", "Elite University", "Future Innovators"
    ]
    states = ["California", "Texas", "New York", "Florida", "Illinois"]

    data = []
    for i in range(250):  # 250 participants
        data.append({
            "Event": random.choice(events),
            "Day": random.choice(days),
            "College": random.choice(colleges),
            "State": random.choice(states)
        })
    df = pd.DataFrame(data)

    # ---------------------------
    # Sidebar: Enhanced Interactive Controls
    # ---------------------------
    st.sidebar.header("Filter Participation Data")
    # Filter selections
    selected_events = st.sidebar.multiselect(
        "Select Event(s):", options=sorted(df["Event"].unique()), default=list(df["Event"].unique())
    )
    selected_days = st.sidebar.multiselect(
        "Select Day(s):", options=sorted(df["Day"].unique()), default=list(df["Day"].unique())
    )
    selected_colleges = st.sidebar.multiselect(
        "Select College(s):", options=sorted(df["College"].unique()), default=list(df["College"].unique())
    )
    selected_states = st.sidebar.multiselect(
        "Select State(s):", options=sorted(df["State"].unique()), default=list(df["State"].unique())
    )
    # Additional sidebar control: Option to show dataset overview
    show_overview = st.sidebar.checkbox("Show Data Overview", value=True)
    # Sidebar control to adjust number of rows to display in overview
    num_rows = st.sidebar.slider("Number of Rows in Overview", min_value=5, max_value=50, value=10, step=5)

    # Filter the DataFrame based on selections
    filtered_df = df[
        (df["Event"].isin(selected_events)) &
        (df["Day"].isin(selected_days)) &
        (df["College"].isin(selected_colleges)) &
        (df["State"].isin(selected_states))
    ]
    st.write(f"Displaying data for **{len(filtered_df)}** participants.")

    # ---------------------------
    # Data Overview Section (Gist)
    # ---------------------------
    if show_overview:
        st.markdown("#### Data Overview")
        # Show sample data in a styled table
        st.dataframe(filtered_df.head(num_rows))
        # Display summary metrics
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total Participants", len(filtered_df))
        colB.metric("Unique Events", filtered_df["Event"].nunique())
        colC.metric("Unique Colleges", filtered_df["College"].nunique())
        colD.metric("Unique States", filtered_df["State"].nunique())
        st.markdown("---")

    # ---------------------------
    # Visualization 1: Event-wise Participation
    # ---------------------------
    st.subheader("Event-wise Participation")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    event_counts = filtered_df["Event"].value_counts().reset_index()
    event_counts.columns = ["Event", "Count"]
    sns.barplot(data=event_counts, x="Event", y="Count", palette="viridis", ax=ax1)
    ax1.set_xlabel("Event")
    ax1.set_ylabel("Number of Participants")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig1)

    # ---------------------------
    # Visualization 2: Day-wise Participation
    # ---------------------------
    st.subheader("Day-wise Participation")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # Order days numerically (Day 1, Day 2, ...)
    day_order = sorted(filtered_df["Day"].unique(), key=lambda x: int(x.split()[1]))
    day_counts = filtered_df["Day"].value_counts().reindex(day_order).reset_index()
    day_counts.columns = ["Day", "Count"]
    sns.lineplot(data=day_counts, x="Day", y="Count", marker="o", ax=ax2)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Number of Participants")
    st.pyplot(fig2)

    # ---------------------------
    # Visualization 3: College-wise Participation (Top 10)
    # ---------------------------
    st.subheader("College-wise Participation (Top 10)")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    college_counts = filtered_df["College"].value_counts().head(10).reset_index()
    college_counts.columns = ["College", "Count"]
    sns.barplot(data=college_counts, x="College", y="Count", palette="magma", ax=ax3)
    ax3.set_xlabel("College")
    ax3.set_ylabel("Number of Participants")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig3)

    # ---------------------------
    # Visualization 4: State-wise Participation
    # ---------------------------
    st.subheader("State-wise Participation")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    state_counts = filtered_df["State"].value_counts().reset_index()
    state_counts.columns = ["State", "Count"]
    sns.barplot(data=state_counts, x="State", y="Count", palette="coolwarm", ax=ax4)
    ax4.set_xlabel("State")
    ax4.set_ylabel("Number of Participants")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig4)

    # ---------------------------
    # Visualization 5: Heatmap of Event vs Day Participation
    # ---------------------------
    st.subheader("Heatmap: Event vs Day Participation")
    crosstab = pd.crosstab(filtered_df["Event"], filtered_df["Day"])
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu", ax=ax5)
    ax5.set_xlabel("Day")
    ax5.set_ylabel("Event")
    st.pyplot(fig5)

    st.success("Interactive analysis complete!")

# 📸 Gallery Page
elif page == "📸 Gallery":
    st.title("📸 INBLOOM ‘25 - Event Gallery")

    # ---------------------------
    # Day-wise Image Gallery (Dummy Images)
    # ---------------------------
    st.subheader("📅 Event Highlights - Day-wise Gallery")
    day_images = {
        "Day 1": ["day1_image1.jpg", "day1_image2.jpg"],
        "Day 2": ["day2_image1.jpg", "day2_image2.jpg"],
        "Day 3": ["day3_image1.jpg", "day3_image2.jpg"],
        "Day 4": ["day4_image1.jpg", "day4_image2.jpg"],
        "Day 5": ["day5_image1.jpg", "day5_image2.jpg"],
    }

    selected_day = st.selectbox("Select Day:", list(day_images.keys()))
    cols = st.columns(3)
    
    for idx, img_name in enumerate(day_images[selected_day]):
        with cols[idx % 3]:
            st.image(f"dummy_images/{img_name}", caption=f"{selected_day} - Image {idx+1}", use_column_width=True)

    # ---------------------------
    # Upload Image Feature
    # ---------------------------
    st.subheader("📤 Upload & Process Event Photos")
    uploaded_images = st.file_uploader("Upload event photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_images:
        cols = st.columns(3)

        for index, img_file in enumerate(uploaded_images):
            with cols[index % 3]:
                image = Image.open(img_file)
                st.image(image, caption="Original Image", use_column_width=True)

                # Image Processing Options
                st.markdown("**🛠 Apply Image Processing:**")
                rotation_angle = st.slider(f"Rotate Image {index+1} (°)", -180, 180, 0)
                resize_width = st.slider(f"Resize Width {index+1}", 50, 500, image.width)
                resize_height = st.slider(f"Resize Height {index+1}", 50, 500, image.height)
                contrast_factor = st.slider(f"Contrast {index+1}", 0.5, 3.0, 1.0)
                brightness_factor = st.slider(f"Brightness {index+1}", 0.5, 3.0, 1.0)
                apply_blur = st.checkbox(f"Apply Gaussian Blur {index+1}")
                apply_color_change = st.checkbox(f"Random Color Shift {index+1}")

                # Convert PIL Image to OpenCV format (NumPy array)
                img_np = np.array(image)

                # Apply Transformations
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE) if rotation_angle == 90 else img_np
                img_np = cv2.resize(img_np, (resize_width, resize_height))

                # Contrast & Brightness
                img_np = cv2.convertScaleAbs(img_np, alpha=contrast_factor, beta=brightness_factor * 50)

                if apply_blur:
                    img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

                if apply_color_change:
                    img_np = np.clip(img_np * np.random.uniform(0.8, 1.2, img_np.shape), 0, 255).astype(np.uint8)

                # Convert Back to PIL for Display
                processed_img = Image.fromarray(img_np)

                # Display Processed Image
                st.image(processed_img, caption="Processed Image", use_column_width=True)

                # ---------------------------
                # 📥 Save & Download Processed Image using OpenCV
                # ---------------------------
                save_path = f"processed_{index+1}.png"
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))  # Save using OpenCV

                with open(save_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=file,
                        file_name=f"processed_{index+1}.png",
                        mime="image/png"
                    )

    st.info("✨ Upload event moments, enhance them, and share the experience! 📸")

# Footer
st.sidebar.markdown("**Developed by 🚀 INBLOOM ‘25 Website Team**")
