import streamlit as st                  # pip install streamlit
import streamlit as st
from PIL import Image

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.markdown("#### Description")
st.markdown("""Build a machine-learning pipeline that can predict the salaries of NBA players based on their performance and background. 
While many players hire agents to negotiate their contracts, there are still cases where players receive salaries that do not match their performance. 
""")
st.markdown("Our hope with this project is to provide players and agents with a tool that can help them estimate their salaries more accurately.")

image = Image.open('NBAimage.jpg')

st.image(image)
st.write("Image credit:")
st.write("URL: https://nba.nbcsports.com/2015/11/10/dwyane-wade-kobe-bryant-is-the-greatest-player-of-our-era/")
st.write("Posted By Dan Feldman")

st.markdown('#### Dataset')
st.write("[NBA Players & Team Data](https://www.kaggle.com/datasets/loganlauton/nba-players-and-team-data?select=NBA+Player+Stats%281950+-+2022%29.csv)")
st.write("[NBA Players stats since 1950](https://www.kaggle.com/datasets/drgilermo/nba-players-stats?select=Seasons_Stats.csv)")

st.markdown("#### Team")
st.markdown("Jiarong Chen (jc2924)")
st.markdown("Yufei Wang (yw569)")

st.markdown("Click **Explore Dataset** to get started.")
