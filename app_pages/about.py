import streamlit as st

def show():
    st.title("🔬 Abstract & Research Team")
    

    st.markdown("### Abstract")

    st.html("""
            <div style='text-align: left;'>
            
            <p>
            Intracranial hemorrhage detection (ICH) is a serious medical condition that
            necessitates a prompt and exhaustive medical diagnosis. This project presents a hybrid deep learning approach that combines Convolutional Neural
            Networks (CNN) and Long Short Term Memory approaches (LSTM). The model achieved a private
            leaderboard log loss of 0.04604 on the RSNA 2019 ICH Detection Challenge, demonstrating
            strong generalization to unseen data.
            
            </p>
            
            <hr style="border: 1px solid #ccc;"/>
            
            </div>
            """)
    
    st.markdown("### 👨‍🔬 Meet the Research Team")
    
    st.html("""
    
    <style>
        .author-container {
            display: flex;
            justify-content: center;
            gap: 80px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .author-card {
            text-align: center;
            font-family: 'Arial', sans-serif;
            padding: 10px;
        }
        .author-img {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #ddd;
            margin-bottom: 10px;
        }
        .author-name {
            font-size: 20px;
            margin: 5px 0;
        }
        .author-links a {
        margin: 0 10px;
        display: inline-block;
    }
    
    </style>

    <div class="author-container">
        <div class="author-card">
            <img src="https://i.postimg.cc/8zG4f3gZ/me.png" class="author-img" alt="Ahmad M">
            <p class="author-name" style="font-weight: bold;">Ahmad M. Alqaisi</p>
            <p style="font-size: 14px; color: #555; margin-top: 20px; margin-bottom: 5px; line-height: 1.2;">
                Data Science and AI Graduate from Al Albayt University<br>
                Medical Imaging • AI Research • Deep Learning Enthusiast
            </p>
            <div class="author-links">
                <a href="mailto:ahmad.alqaisi215@gmail.com" title="Email Ahmad">📧</a>
                <a href="https://www.linkedin.com/in/ahmad-al-qaisi/" target="_blank" title="LinkedIn">🔗</a>
            </div>
        </div>
        <div class="author-card">
            <img src="https://i.postimg.cc/J0rRbMPf/IMG-0068-1.jpg" class="author-img" alt="Sarah Al">
            <p class="author-name" style="font-weight: bold;">Sarah K. Almashagbeh</p>
            <p style="font-size: 14px; color: #555; margin-top: 20px; margin-bottom: 5px; line-height: 1.2;">
                Data Science and AI Graduate from Al Albayt University<br>
                Medical Imaging • AI Research • Deep Learning Enthusiast
            </p>
            <div class="author-links">
                <a href="mailto:sarah.almashagbehh@gmail.com" title="Email Sarah">📧</a>
                <a href="https://www.linkedin.com/in/sarah-almashagbeh/" target="_blank" title="LinkedIn">🔗</a>
            </div>
        </div>
    </div>
    """)
    
    st.markdown("---")
    st.markdown("### 📂 Project Access")
    col1, col2 = st.columns(2)

    with col1:
        st.link_button("🔗 View on GitHub", "https://github.com/AMQ4/Automated-Brain-Hemorrhage-Detection-in-CT-Scans")
    with col2:
        with open("ICH_Project_DOC.pdf", "rb") as file:
            st.download_button(
            label="📄 Download Documentation",
            data=file,
            file_name="ICH_Project_DOC.pdf",
            mime="application/pdf"
        )
    
    st.markdown("---")
    #Footer
    st.html("""
            <p style='text-align: center; font-size: 14px; color: gray; margin-top: 30px;'>
            © 2025 Sarah K. Almashagbeh & Ahmad M. Alqaisi – All rights reserved.<br>
            This project was developed as part of our graduation thesis at Al Albayt University.<br>
            For educational and research purposes only.
            </p>
            """)
