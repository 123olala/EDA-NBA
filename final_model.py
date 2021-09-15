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
import joblib
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

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

#Category position
def check_list(li):
    if len(li) != 9:
        return False
    if li[-1] > 3:
        return False
    for i in li:
        if (i > 10) or (i < 1):
            return False
    return True

#Hypothesis
def hypothesis_score_by_position(df):
    hypo_df = df[(df['G'] >= 27) & (df['MP'] >= 12) & (df['Tm'] != 'TOT')]
    pos_mapping = {'C':'frontcourt','PF':'frontcourt','SF':'frontcourt','PG':'backcourt','SG':'backcourt'}
    hypo_df.loc[:,'Pos'] = hypo_df['Pos'].map(pos_mapping)
    
    front_score = hypo_df.loc[hypo_df['Pos'] == 'frontcourt']['PTS']
    back_score = hypo_df.loc[hypo_df['Pos'] == 'backcourt']['PTS']

    statistics,p_val = ttest_ind(front_score, back_score,alternative='less')
    return front_score.mean(), back_score.mean(), statistics, p_val

def hypothesis_3Pscore_by_position(df):
    hypo_df = df[(df['G'] >= 27) & (df['MP'] >= 12) & (df['Tm'] != 'TOT')]
    pos_mapping = {'C':'frontcourt','PF':'frontcourt','SF':'frontcourt','PG':'backcourt','SG':'backcourt'}
    hypo_df.loc[:,'Pos'] = hypo_df['Pos'].map(pos_mapping)
    
    front_score = hypo_df.loc[hypo_df['Pos'] == 'frontcourt']['3P%']
    back_score = hypo_df.loc[hypo_df['Pos'] == 'backcourt']['3P%']

    statistics,p_val = ttest_ind(front_score, back_score,alternative='less')
    return front_score.mean(), back_score.mean(), statistics, p_val
    

#Frontend Content
#Header
st.title('Applied Data Science on Sport: NBA Regular Season Statistics') #Name of web app
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
This project performs simple web scraping, data aggregation and cleaning, data analysis visualization, and hypothesizing of NBA player stats data. Then building a machine learning model to predict the salary of NBA player and classify players position!\n
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
st.subheader('0. Player Position Clustering Visualization using t-SNE')
if st.button('Show t-SNE representation'):
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
st.subheader('1. Correlation matrix')
corr_matrix = playerstats.corr()
fig,ax = themes_plot((12,6))
sns.heatmap(corr_matrix,yticklabels=False,cbar=True,cmap='Blues')
st.pyplot(fig)

#Lineplot
st.subheader('2. NBA gameplay trend')
trend = pd.read_csv('trend.csv')
fig,ax = themes_plot()
sns.lineplot(x='Year',y='2PA',data = trend,color=red,label='2 Point')
sns.lineplot(x='Year',y='3PA',data = trend,color=green,label='3 Point')
ax.set_xlabel('Year')
ax.set_ylabel('Attempts Per Game by a team')
plt.legend()
st.pyplot(fig)

#Histplot
st.subheader('3. Distribution of all attributes')
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
st.subheader('4. Does a good scorer necessarily have to be a good 3-point maker?')
fig,ax = visualize_top20point()

st.subheader('5. Which is the youngest team in the NBA?')
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
st.subheader('6. Explicit team analysis')
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
st.subheader('7. Does player salary depend on age?')
salary_age = playerstats.groupby('Age')['Salary ($)'].mean()
fig,ax = themes_plot()
sns.lineplot(x=salary_age.index,y=salary_age.values,color=np.random.choice([red,green]))
ax.set_ylabel('Salary')
st.pyplot(fig)

#Countplot
st.subheader('8. Player position ratio')
fig,ax = themes_plot()
sns.countplot(x='Pos',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('')
st.pyplot(fig)

#Boxplot
st.subheader('9. Does player salary depend on their position?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='Salary ($)',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
st.pyplot(fig)

#Boxplot
st.subheader('10. Do forwards make more points than guards?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='PTS',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('Point Per Game')
st.pyplot(fig)

#Scatterplot
st.subheader('11. Is a good offensive rebound player good at defensive rebound?')
fig,ax = themes_plot()
sns.scatterplot(x='ORB',y='DRB',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Offensive Rebound')
ax.set_ylabel('Defensive Rebound')
st.pyplot(fig)

#Scatterplot
st.subheader('12. Is a good steal player good at block?')
fig,ax = themes_plot()
sns.scatterplot(x='STL%',y='BLK%',data=playerstats,hue='Pos',palette='Set2')
ax.set_xlabel('Steal percentage contribution')
ax.set_ylabel('Block percentage contribution')
st.pyplot(fig)

#Scatterplot
st.subheader('13. Box Plus Minus (BPM) versus Value Over Replacement Player (VORP)?')
fig,ax = themes_plot()
sns.scatterplot(x='BPM',y='VORP',data=playerstats,hue='Pos',palette='Set2')
st.pyplot(fig)

#Scatterplot
st.subheader('14. How Point Per Game affects Winshare (WS), Box Plus Minus (BPM), Value Over Replacement Player (VORP)?')

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
st.subheader('15. Are Point-Guards the best playmarker?')
fig,ax = themes_plot()
sns.boxplot(x='Pos',y='AST%',data=playerstats,palette='Set2')
ax.set_xlabel('Position')
ax.set_ylabel('Assist percentage contribution')
st.pyplot(fig)

#5. Hypothesis testing
st.header('V. Hypothesis Testing')
# st.code('''"coming soon..."''',language='python')
hypo_df = playerstats[(playerstats['G'] >= 27) & (playerstats['MP'] >= 12) & (playerstats['Tm'] != 'TOT')]
#5.1
st.subheader('1. Do guards score more than forwards on average?')
st.code('''H0: mean_score(forwards) >= mean_score(guards)\
    \n\
H1: mean_score(forwards) < mean_score(guards)''',language='python')
forwards_score, guards_score, statistics, p_val = hypothesis_score_by_position(hypo_df)
st.write(f'- Mean score of forwards (Observation): {np.round(forwards_score,2)}')
st.write(f'- Mean score of guards (Observation): {np.round(guards_score,2)}')
st.write(f'- p value: {p_val}')
if p_val <= 0.05:
    st.write("At 5% significance, we reject the null hypothesis.")
    st.write('=> Guards are significantly better at scoring than Forwards.')
else:
    st.write("At 5% significance, we cannot reject the null hypothesis.")
    st.write('=> There is not enough statistical evidence that Guards are significantly better at scoring than Forwards.')
#5.2
st.subheader('2. Do guards make 3-point more accurate than forwards on average?')
st.code('''H0: mean_3P_accuracy(forwards) >= mean_3P_accuracy(guards)\
    \n\
H1: mean_3P_accuracy(forwards) < mean_3P_accuracy(guards)''',language='python')
forwards3P_accuracy, guards3P_accuracy, statistics, p_val = hypothesis_3Pscore_by_position(hypo_df)
st.write(f'- Mean 3-point accuracy of forwards (Observation): {np.round(forwards3P_accuracy,2)}')
st.write(f'- Mean 3-point accuracy of guards (Observation): {np.round(guards3P_accuracy,2)}')
st.write(f'- p value: {p_val}')
if p_val <= 0.05:
    st.write("At 5% significance, we reject the null hypothesis.")
    st.write('=> Guards are significantly better at scoring 3-point than Forwards.')
else:
    st.write("At 5% significance, we cannot reject the null hypothesis.")
    st.write('=> There is not enough statistical evidence that Guards are significantly better at scoring 3-point than Forwards.')
#5.3
st.subheader('3.  Does a good scorer is also good at scoring 3-point?')
st.code('''H0: 3-point accuracy and scoring ability are independent.\
    \n\
H1: 3-point accuracy and scoring ability are dependent.''',language='python')
correlation,p_val = pearsonr(hypo_df['PTS'], hypo_df['3P%'])
st.write(f'- Correlation: {correlation}')
st.write(f'- p value: {p_val}')
if p_val <= 0.05:
    if correlation >= 0.2:
        st.write("=> At 5% significance, there is a relationship between 3-point accuracy and scoring ability.")
    else:
        st.write("=> At 5% significance, there is a very week relationship between 3-point accuracy and scoring ability.")
else:
    st.write("=> At 5% significance, we cannot conclude that there is a relationship between 3-point accuracy and scoring ability.")
#5.4
st.subheader('4. Does the salary of players really dependent on their effectiveness on the court?')
st.code('''H0: Salary and player effectiveness are independent.\
    \n\
H1: Salary and player effectiveness are dependent.''',language='python')
correlation,p_val = pearsonr(hypo_df.dropna()['BPM'], hypo_df.dropna()['Salary ($)'])
st.write(f'- Correlation: {correlation}')
st.write(f'- p value: {p_val}')
if p_val <= 0.05:
    if correlation >= 0.2:
        st.write("=> At 5% significance, there is a relationship between Salary and player effectiveness.")
    else:
        st.write("=> At 5% significance, there is a very week relationship between Salary and player effectiveness.")
else:
    st.write("=> At 5% significance, we cannot conclude that there is a relationship between Salary and player effectiveness.")

#6. Machine Learning: Predict salary
st.header('VI. Machine Learning: (FOR FUNNY)')

#Model
scaler_clf = joblib.load('scaler_clf.sav')
k_nearest = joblib.load('k_nearest.sav')
rf_clf = joblib.load('rf_clf.sav')
svc = joblib.load('svc_clf.sav')
xgb_clf = joblib.load('xgb_clf.sav')
rf_reg = joblib.load('rf_reg.sav')
scaler_reg = joblib.load('scaler_reg.sav')

dict_model = {'XGBoost':xgb_clf, 'Support Vector Machine':svc, 'Random Forest':rf_clf, 'k-Nearest Neighbor':k_nearest}
dict_position = {'PG':'Point Guard', 'SG':'Shooting Guard','SF':'Small Forward','PF':'Power Forward','C':'Center'}
df_quest = pd.DataFrame(['What is your scoring ability? (from 1 to 10)',\
    'What is your shooting accuracy? (from 1 to 10)',\
        'What is your shooting frequency? (from 1 to 10)',\
            'What is your 3-point shooting ability? (from 1 to 10)',\
                'What is your rebound ability? (from 1 to 10)',\
                    'What is your assist ability? (from 1 to 10)',\
                        'What is your steal ability? (from 1 to 10)',\
                            'What is your block ability? (from 1 to 10)',\
                                'How do you feel about your height? (from 1 to 3)'])
df_quest.columns = ['Provided Information']
st.dataframe(df_quest)
#Input user
input_user = st.text_input('Provided information split by commas')
att_user = input_user.split(',')

st.subheader("1. What is your position if you play basketball?")
model_list = ['XGBoost','Support Vector Machine','Random Forest','k-Nearest Neighbor']
model = dict_model[st.selectbox('Choose algorithm:',model_list)]
if st.button("*Oki let's discover your position*"):
    if len(input_user) != 0:
        try:
            att_num_user = [int(i) for i in att_user]
            if not check_list(att_num_user):
                st.write('**Wrong User Input**')
            else:
                user_position = model.predict(scaler_clf.transform(np.array(att_num_user).reshape(1,-1)))
                st.write(f'You seem like a good fit for **{dict_position[user_position[0]]}**.')
        except:
            st.write('**Wrong User Input**')      
    else:
        st.write('**Wrong User Input**')

st.subheader("2. What is your salary if you compete in NBA?")
if st.button("*Oki let's predict your salary*"):
    if len(input_user) != 0:
        try:
            att_num_user = [int(i) for i in att_user]
            if not check_list(att_num_user):
                st.write('**Wrong User Input**')
            else:
                user_salary = rf_reg.predict(scaler_reg.transform(np.array(att_num_user).reshape(1,-1)))
                st.write(f'You maybe receive **{np.round(user_salary[0]/1000000,3)} millions $** when compete in NBA.')
        except:
            st.write('**Wrong User Input**')
    else:
        st.write('**Wrong User Input**')
