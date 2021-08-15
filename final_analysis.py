import streamlit as st 
import pandas as pd 
import base64 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import matplotlib.image as img 
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import unicodedata
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

#Tune the visualization
pd.set_option('display.precision', 2)
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['figure.dpi'] = 150
red = '#B73832'
green = '#244747'
background_color='#F5F4EF'
def themes_plot(figsize=(12,6)):
    #red = '#B73832'
    #green = '#244747'
    background_color='#F5F4EF'
    fig,ax = plt.subplots(figsize=figsize,facecolor=background_color)
    ax.set_facecolor(background_color)
    ax.tick_params(axis=u'both',which=u'both',length=0)
    
    return fig,ax

#Disable warnings
import warnings
warnings.filterwarnings('ignore')

#Backend Content

#Web scraping of NBA player stats 
@st.cache 
def load_pergame(year):
    """ Return DataFrame of NBA Player Stats: Per Game """ 

    url_per = 'https://www.basketball-reference.com/leagues/NBA_' \
        +str(year) + '_per_game.html'
    html_per = pd.read_html(url_per, header = 0)
    df_per = html_per[0]
    raw_per = df_per.drop(df_per[df_per.Age == 'Age'].index)  # Deletes repeating headers in content
    raw_per = raw_per.fillna(0) 
    per = raw_per.drop(['Rk'],axis=1)  #Deletes columns 'Rk'
    per['Player'] = per['Player'].str.replace('.','',regex=False)
    per['Player'] = per['Player'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
    per['Player'] = per['Player'].str.strip()

    return per

def load_advanced(year):
    """ Return DataFrame of NBA Player Stats: Advanced """ 

    url_ad = 'https://www.basketball-reference.com/leagues/NBA_' \
        +str(year) + '_advanced.html'
    html_ad = pd.read_html(url_ad, header = 0)
    df_ad = html_ad[0]
    raw_ad = df_ad.drop(df_ad[df_ad.Age == 'Age'].index)    # Deletes repeating headers in content
    raw_ad = raw_ad.fillna(0) 
    ad = raw_ad.drop(['Rk','Pos','Age','G','MP','Unnamed: 19','Unnamed: 24'],axis=1)
    ad['Player'] = ad['Player'].str.replace('.','',regex=False)
    ad['Player'] = ad['Player'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
    ad['Player'] = ad['Player'].str.strip()

    return ad

def load_possessions(year):
    """ Return DataFrame of NBA Player Stats: Per 100 Possessions """ 

    url_poss = 'https://www.basketball-reference.com/leagues/NBA_' \
        +str(year) + '_per_poss.html'
    html_poss = pd.read_html(url_poss, header = 0)
    df_poss = html_poss[0]
    raw_poss = df_poss.drop(df_poss[df_poss.Age == 'Age'].index)    # Deletes repeating headers in content
    raw_poss = raw_poss.fillna(0) 
    poss = raw_poss[['Player','Tm','ORtg','DRtg']]
    poss['Player'] = poss['Player'].str.replace('.','',regex=False)
    poss['Player'] = poss['Player'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
    poss['Player'] = poss['Player'].str.strip()

    return poss

def load_salary(year):
    """ Return DataFrame of NBA Player's Salary """ 

    url = 'https://hoopshype.com/salaries/players/' + str(year-1) + '-' \
        + str(year)
    salary_col = str(year-1) + '/' + str(year)[-2:]

    def convert_salary(str):        #Convert $money into number
        string = str[1:]        #Remove character "$"
        salary = string.split(',')       #Remove character "," 
        salary_num = int(''.join(salary))       #Convert string into number

        return salary_num

    html = pd.read_html(url)
    df = html[0]
    salary = df[['Player',salary_col]]
    salary_num = [convert_salary(i) for i in salary[salary_col]]        #Get list of salary number
    salary['Salary ($)'] = salary_num       #Create Salary column
    salary = salary.drop([salary_col],axis=1)       #Drop column of $money
    salary['Player'] = salary['Player'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
    salary['Player'] = salary['Player'].str.strip()

    return salary

def load_data(year):

    def convert_position(pos):
        if '-' in pos:
            pos = pos.split('-')[0]
        elif pos == 'G':
            pos = 'PG'
        elif pos == 'F':
            pos = 'PF'
        else:
            pos = pos 

        return pos

    playerstats = load_pergame(year).merge(load_advanced(year),how='inner')
    playerstats = playerstats.merge(load_possessions(year),how='inner')
    if year > 1990:
        playerstats = playerstats.merge(load_salary(year),on='Player',how='left')
    
    pos_value = np.array([convert_position(pos) for pos in playerstats['Pos']])
    playerstats['Pos'] = pos_value

    spec_chars = ["!",'"',"#","%","&","'","(",")", \
    "*","+",",","-",".","/",":",";","<", \
        "=",">","?","@","[","\\","]","^","_", \
            "`","{","|","}","~","–"]
    for char in spec_chars:
        playerstats['Player'] = playerstats['Player'].str.replace(char, ' ')
    playerstats['Player'] = playerstats['Player'].str.split().str.join(" ")
    
    return playerstats

#Correct dataframe type (for visualization)
def correct_df(df):
    object_type = ['Player','Pos','Tm']
    for col in df.drop(object_type,axis=1).columns:
        df[col] = df[col].astype(float)

    return df

#NBA top 20
def NBA_top20(label):
    if label == 'PER':
        cond = (playerstats['GS'] >= 41) & (playerstats['Tm'] != 'TOT')
        df_20 = playerstats[cond].sort_values(label,ascending=False)[['Player','Pos','Tm',label,'FG%','3P%','FT%','Salary ($)']].head(20)
        return df_20
    elif label == 'Salary ($)':
        cond = playerstats['Tm'] != 'TOT'
        df_20 = playerstats[cond].drop_duplicates('Player').sort_values(label,ascending=False)[['Player','Pos','Tm',label]].head(20)
        return df_20
    cond = (playerstats['G'] >= 27) & (playerstats['MP'] >= 12) & (playerstats['Tm'] != 'TOT')
    df_20 = playerstats[cond].sort_values(label,ascending=False)[['Player','Pos','Tm',label,'FG%','3P%','FT%','Salary ($)']].head(20)

    return df_20

#Download CSV
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() #string <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download summary statistics</a>'
    return href

#Explicit team plot
def histplot_team(Tm):
    df_plot = playerstats[playerstats['Tm'] == Tm]
    mean = round(df_plot['PTS'].mean(),2)
    std = round(df_plot['PTS'].std(),2)
    fig, ax = themes_plot()
    sns.histplot(x=df_plot['PTS'],color=np.random.choice([red,green]),kde=True,label=f'mean = {mean}\nstd = {std}')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Point Per Game Distribution')
    ax.legend()
    st.pyplot(fig)

    return fig,ax

def barplot_team(Tm):
    df_plot = playerstats[playerstats['Tm'] == Tm]
    df_plot['FTS'] = df_plot['FT']
    df_plot['2PS'] = df_plot['2P'] * 2 
    df_plot['3PS'] = df_plot['3P'] * 3 

    fig, ax = themes_plot()
    ax = plt.barh(df_plot['Player'],df_plot['FTS'],label='Free Throw',color='#66c2a5')
    ax = plt.barh(df_plot['Player'],df_plot['2PS'],left=df_plot['FTS'],label='2 Point',color='#fc8d62')
    ax = plt.barh(df_plot['Player'],df_plot['3PS'],left=df_plot['2PS']+df_plot['FTS'],label='3 Point',color='#8da0cb')
    plt.xlabel('Point Per Game')
    plt.legend()
    st.pyplot(fig)

    point = df_plot[df_plot['PTS'] >= 15]['PTS']
    star = df_plot[df_plot['PTS'] >= 15]['Player']
    return star,len(star),point

def visualize_top20point(label='PTS'):
    cond = (playerstats['G'] >= 27) & (playerstats['MP'] >= 12) & (playerstats['Tm'] != 'TOT')
    score_20 = playerstats[cond].sort_values(label,ascending=False)[['Player','Pos','Tm',label,'FT','2P','3P']].head(20)
    score_20['FTS'] = score_20['FT']
    score_20['2PS'] = score_20['2P'] * 2 
    score_20['3PS'] = score_20['3P'] * 3

    fig, ax = themes_plot()
    ax = plt.barh(score_20['Player'],score_20['FTS'],label='Free Throw',color='#66c2a5')
    ax = plt.barh(score_20['Player'],score_20['2PS'],left=score_20['FTS'],label='2 Point',color='#fc8d62')
    ax = plt.barh(score_20['Player'],score_20['3PS'],left=score_20['2PS']+score_20['FTS'],label='3 Point',color='#8da0cb')
    plt.xlabel('Point Per Game')
    plt.title('Top 20 Best NBA Scorers')
    plt.legend()
    st.pyplot(fig)

    return fig,ax

#Scale data to range 10
def scale_10(num,df,label):
    if num < df[label].quantile(0.1):
        return 1
    elif num < df[label].quantile(0.2):
        return 2
    elif num < df[label].quantile(0.3):
        return 3
    elif num < df[label].quantile(0.4):
        return 4
    elif num < df[label].quantile(0.5):
        return 5
    elif num < df[label].quantile(0.6):
        return 6
    elif num < df[label].quantile(0.7):
        return 7
    elif num < df[label].quantile(0.8):
        return 8
    elif num < df[label].quantile(0.9):
        return 9
    return 10

#Frontend Content
#Header
st.title('NBA Player Stats Data Exploratory') #Name of web app
st.code('''"made by cao long ạ"\n"DOB: 20/11/2001"\n"Univeristy of Economics and Law"\n"Class: 14"''',language='python') #Introduction
background = Image.open('nba.jpeg')
st.image(background,caption='NBA Allstar')

#Body

#Sidebar
st.sidebar.header('Data Filtering') #Name of the sidebar header

#Year selection
selected_year = st.sidebar.selectbox('Year',list(reversed(range(1950,2022)))) 
playerstats = load_data(selected_year) #Final summary statistics after scraping
playerstats = correct_df(playerstats)
#Team selection 
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team',sorted_unique_team,sorted_unique_team)
#Position selection
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position',unique_pos,unique_pos)
#Age selection
unique_age = sorted(playerstats.Age.unique())
selected_age = st.sidebar.multiselect('Age',unique_age,unique_age)
#Player name selection
selected_name = st.sidebar.text_input('Player name')

#Main page
#1.Introduction
st.header('I. Introduction')
st.markdown("""
This project performs simple web scraping, data aggregation and cleaning, data analysis visualization, and hypothesizing of NBA player stats data. Then building a machine learning model to predict the salary of NBA player!\n
* **Python libraries:** pandas, matplotlib, seaborn, numpy, sklearn, scipy\
    , streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/),\
     [Hoopshype.com](https://hoopshype.com/salaries/)
""")

#2.Summary statistics
st.header('II. Summary statistics of player')
#Filtering data
df_selected = playerstats[(playerstats.Tm.isin(selected_team)) \
    & (playerstats.Pos.isin(selected_pos)) \
        & (playerstats.Age.isin(selected_age))]
if len(selected_name) > 0:
    df_selected = df_selected[df_selected['Player'].str.lower().str.contains(selected_name.lower().strip())]
#Display Stats
# df_selected = correct_df(df_selected)
st.write('Data Dimension: ' + str(df_selected.shape[0]) \
    + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
st.dataframe(df_selected)
#Where users can download the data
st.markdown(file_download(df_selected),unsafe_allow_html=True)

#3. NBA top 20
st.header('III. NBA Top 20')

st.subheader('1. Point Per Game Leader - Top 20')
st.dataframe(NBA_top20('PTS'))
st.subheader('2. Rebound Per Game Leader - Top 20')
st.dataframe(NBA_top20('TRB'))
st.subheader('3. Assist Per Game Leader - Top 20')
st.dataframe(NBA_top20('AST'))
st.subheader('4. 3-Point Per Game Leader - Top 20')
st.dataframe(NBA_top20('3P'))
st.subheader('5. Steal Per Game Leader - Top 20')
st.dataframe(NBA_top20('STL'))
st.subheader('6. Block Per Game Leader - Top 20')
st.dataframe(NBA_top20('BLK'))
st.subheader('7. Efficiency Player Leader - Top 20')
st.dataframe(NBA_top20('PER'))
if selected_year >= 1990:
    st.subheader('8. Top 20 Highest Salary Player')
    st.dataframe(NBA_top20('Salary ($)'))

#4. Data analysis and visualization
st.header('IV. Data analysis and visualization')

#TSNE clustering
st.subheader('1. Position Clustering Visualization using t-SNE')
tnse_df = playerstats[(playerstats['G']>15) & (playerstats['MP']>5)][['Player','Salary ($)','PTS','Pos','FG%','FGA','3P%','FT%','TRB%','AST%','STL%','BLK%']]
position_mapping = {'PG':'Guard','SG':'Guard','SF':'Forward','PF':'Forward','C':'Center'}
tnse_df['Pos'] = tnse_df['Pos'].map(position_mapping)
tsne_att = tnse_df.drop(['Player','Salary ($)','PTS','Pos'],axis=1)
for column in tsne_att.columns:
    tsne_att[column] = tsne_att[column].apply(scale_10,args=(tsne_att,column))
tsne = TSNE(random_state=3107)
tsne_rep = tsne.fit_transform(tsne_att)
fig,ax = themes_plot()
sns.scatterplot(x=tsne_rep[:,0],y=tsne_rep[:,1],hue=tnse_df['Pos'],palette='Set2')
st.pyplot(fig)

#Correlation matrix
st.subheader('2. Correlation matrix')
corr_matrix = playerstats.corr()
fig,ax = themes_plot((12,6))
sns.heatmap(corr_matrix,yticklabels=False,cbar=True,cmap='Blues')
st.pyplot(fig)

#Lineplot
st.subheader('3. NBA gameplay trend')
trend = pd.read_csv('trend.csv')
fig,ax = themes_plot()
sns.lineplot(x='Year',y='2PA',data = trend,color=red,label='2 Point')
sns.lineplot(x='Year',y='3PA',data = trend,color=green,label='3 Point')
ax.set_xlabel('Year')
ax.set_ylabel('Attempts Per Game by a team')
plt.legend()
st.pyplot(fig)

#Histplot
st.subheader('4. Distribution of all attributes')
stats = list(playerstats.columns)[4:]
stats = stats + ['Age']
selected_stats = st.selectbox('Attribute',stats)
fig,ax = themes_plot()
mean = round(playerstats[selected_stats].mean(),2)
std = round(playerstats[selected_stats].std(),2)
sns.histplot(playerstats[selected_stats],kde=True,color=np.random.choice([red, green]),label=f'mean = {mean}\nstd = {std}')
ax.set_xlabel('')
ax.set_ylabel('')
plt.legend()
st.pyplot(fig)  

#Barplot
st.subheader('5. Does a good scorer necessarily have to be a good 3-point maker?')
fig,ax = visualize_top20point()

st.subheader('6. Which is the youngest team in the NBA?')
nba_age = playerstats[playerstats['Tm'] != 'TOT'].groupby('Tm')['Age'].mean().sort_values()
range_yaxis = range(1,len(nba_age)+1)
fig,ax = themes_plot((12,10))
plt.hlines(y=range_yaxis, xmin=0, xmax=nba_age.values, color='gray')
plt.plot(nba_age.values, range_yaxis, "o",markersize=10, color=np.random.choice([red, green]))
plt.yticks(range_yaxis, nba_age.index)
ax.set_xlabel('Age')
ax.set_ylabel('Team')
st.pyplot(fig)

#Explicit plot
st.subheader('7. Explicit team analysis')
team = sorted(playerstats['Tm'].unique())
selected_team_ex = st.selectbox('Team',team,key='hist')
    #Histplot
fig,ax = histplot_team(selected_team_ex)
    #Pieplot
fig,ax = themes_plot()
df_pie = playerstats[playerstats['Tm'] == selected_team_ex]
sc_pie = {'3 Point Attempts':df_pie['3PA'].sum(),'2 Point Attempts':df_pie['2PA'].sum()}
plt.pie(x=sc_pie.values(), autopct="%.1f%%", explode=[0.05]*2, labels=sc_pie.keys(), pctdistance=0.5,colors=[red,green])
plt.title('Gameplay Strategy')
st.pyplot(fig)
    #Barplot
star,amount,point = barplot_team(selected_team_ex)
if amount > 1:
    st.markdown(f'**=> There are {amount} players scoring more than 15 Point Per Game.**')
else:
    st.markdown(f'**=> There is {amount} player scoring more than 15 Point Per Game.**')

for i in range(amount):
    st.markdown(f'* {star.values[i]} : {point.values[i]} point per game.')

#Lineplot
st.subheader('8. Does player salary depend on age?')
salary_age = playerstats.groupby('Age')['Salary ($)'].mean()
fig,ax = themes_plot()
sns.lineplot(x=salary_age.index,y=salary_age.values,color=np.random.choice([red,green]))
ax.set_ylabel('Salary')
st.pyplot(fig)

#Countplot
st.subheader('9. Player position ratio')
fig,ax = themes_plot()
sns.countplot(x='Pos',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('')
st.pyplot(fig)

#Boxplot
st.subheader('10. Does player salary depend on their position?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='Salary ($)',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
st.pyplot(fig)

#Boxplot
st.subheader('11. Do forwards make more points than guards?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='PTS',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('Point Per Game')
st.pyplot(fig)

#Scatterplot
st.subheader('12. Is a good offensive rebound player good at defensive rebound?')
fig,ax = themes_plot()
sns.scatterplot(x='ORB',y='DRB',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Offensive Rebound')
ax.set_ylabel('Defensive Rebound')
st.pyplot(fig)

#Scatterplot
st.subheader('13. Is a good steal player good at block?')
fig,ax = themes_plot()
sns.scatterplot(x='STL%',y='BLK%',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Steal percentage contribution')
ax.set_ylabel('Block percentage contribution')
st.pyplot(fig)

#Scatterplot
st.subheader('14. Box Plus Minus (BPM) versus Value Over Replacement Player (VORP)?')
fig,ax = themes_plot()
sns.scatterplot(x='BPM',y='VORP',data=playerstats,hue='Pos',palette='Set2')
st.pyplot(fig)

#Scatterplot
st.subheader('15. How Point Per Game affects Winshare (WS), Box Plus Minus (BPM), Value Over Replacement Player (VORP)?')

fig,ax = themes_plot()
sns.scatterplot(x='PTS',y='WS',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Point Per Game')
ax.set_title('Point Per Game vs Winshare')
st.pyplot(fig)

fig,ax = themes_plot()
sns.scatterplot(x='PTS',y='BPM',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Point Per Game')
ax.set_title('Point Per Game vs Box Plus Minus')
st.pyplot(fig)

fig,ax = themes_plot()
sns.scatterplot(x='PTS',y='VORP',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Point Per Game')
ax.set_title('Point Per Game vs Value Over Replacement Player')
st.pyplot(fig)

#Box plot
st.subheader('16. Are Point-Guards the best playmarker?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='AST%',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('Assist percentage contribution')
st.pyplot(fig)
#5. Hypothesis testing
st.header('V. Hypothesis Testing')
st.code('''"coming soon..."''',language='python')
#6. Machine Learning: Predict salary
st.header('VI. Machine Learning: (FOR FUNNY)')
st.code('''"coming soon..."''',language='python')


