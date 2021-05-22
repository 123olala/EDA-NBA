import streamlit as st 
import pandas as pd 
import base64 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 

st.title('NBA Player Stats Explorer') #Name of web app

st.code(''' "made by cao long ạ" ''',language='python')

from PIL import Image #Insert Image
image = Image.open('nba.jpeg')
st.image(image,caption='NBA Allstar')

st.markdown("""
This app performs simple webscraping of NBA player stats data!
* **Python libraries:** base64, pandas, matplotlib, seaborn, numpy, \
    sklearn, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/),\
     [Hoopshype.com](https://hoopshype.com/salaries/)
""")    #Discription

st.sidebar.header('User Input Features') #Name of the sidebar header
selected_year = st.sidebar.selectbox('Year',\
    list(reversed(range(1950,2022)))) #Search year for data 

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
    
    return poss

def load_salary(year):
    """ Return DataFrame of NBA Player's Salary """ 

    url_2021 = 'https://hoopshype.com/salaries/players/'
    url = 'https://hoopshype.com/salaries/players/' + str(year-1) + '-' \
        + str(year)
    salary_col = str(year-1) + '/' + str(year)[-2:]

    def convert_salary(str):        #Convert $money into number
        string = str[1:]        #Remove character "$"
        salary = string.split(',')       #Remove character "," 
        salary_num = int(''.join(salary))       #Convert string into number

        return salary_num

    #Create DataFrame
    if year == 2021:
        html = pd.read_html(url_2021)
        df = html[0]
        salary = df[['Player',salary_col]]
        salary_num = [convert_salary(i) for i in salary[salary_col]]        #Get list of salary number
        salary['Salary ($)'] = salary_num       #Create Salary column
        salary = salary.drop([salary_col],axis=1)       #Drop column of $money
    else:           #if Year != 2021
        html = pd.read_html(url)
        df = html[0]
        salary = df[['Player',salary_col]]
        salary_num = [convert_salary(i) for i in salary[salary_col]]        #Get list of salary number
        salary['Salary ($)'] = salary_num       #Create Salary column
        salary = salary.drop([salary_col],axis=1)       #Drop column of $money
    
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
    
playerstats = load_data(selected_year)

#Sidebar - Team selection 
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team',sorted_unique_team,sorted_unique_team)

#Sidebar - Position selection
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position',unique_pos,unique_pos)

#Sidebar - Age selection
unique_age = sorted(playerstats.Age.unique())
selected_age = st.sidebar.multiselect('Age',unique_age,unique_age)

#Sidebar - Player name selection
selected_name = st.sidebar.text_input('Player name')

#Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) \
    & (playerstats.Pos.isin(selected_pos)) \
        & (playerstats.Age.isin(selected_age))]

if len(selected_name) > 0:
    df_selected_team = df_selected_team[df_selected_team['Player'] == selected_name]

#Display Stats
st.header('Display Player Stats of Selected')
st.write('Data Dimension:' + str(df_selected_team.shape[0]) \
    + 'rows and ' + str(df_selected_team.shape[1]) + 'columns.')
st.subheader('Stats Per Game')
st.dataframe(df_selected_team)

#Download NBA player stats data
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() #string <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(file_download(df_selected_team),unsafe_allow_html=True)

#Individual award
st.subheader('Individual Leader')
def button(label,header,col):
    if st.button(label):
        st.header(header)
        playerstats.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')
        df = df[(df['G'] >= 27) & (df['MP'] >=12)]
        df = df[df['Tm'] != 'TOT'].sort_values(by=[col],ascending=False)
        st.dataframe(df.head(15))

button('PPG Leader','Point Per Game Leader - Top 15','PTS')
button('RPG Leader','Rebound Per Game Leader - Top 15','TRB')
button('APG Leader','Assist Per Game Leader - Top 15','AST')
button('3P Leader','3-Point Per Game Leader - Top 15','3P')
button('STL Leader','Steal Per Game Leader - Top 15','STL')
button('BLK Leader','Block Per Game Leader - Top 15','BLK')
button('WS Leader','Win Share Leader - Top 15','WS')
button('BPM Leader',"Top 15 Best Player's Impact ",'BPM')

if selected_year >= 1995:
    if st.button('3P% Leader'):
        st.header('3-Point-Percentage Per Game Leader - Top 15')
        playerstats.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')
        df = df[(df['3PA'] >= 5) & (df['G'] >= 41)]
        df = df[df['Tm'] != 'TOT'].sort_values(by=['3P%'],ascending=False)
        st.dataframe(df.head(15))

if selected_year > 1990:
    if st.button('Salary Leader'):
        st.header('Salary Leader')
        playerstats.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')
        df = df.sort_values(by=['Salary ($)'],ascending=False)
        st.dataframe(df.head(15))

#Efficiency player calculation
if st.button('EFF Leader'):
        st.header('Efficiency Player Leader - Top 15')
        playerstats.to_csv('output.csv',index=False)
        df = pd.read_csv('output.csv')
        df = df[df['GS'] >= 41]
        df = df.sort_values(by=['PER'],ascending=False)
        st.dataframe(df.head(15))
        
#Best defensive player
if st.button('DEF Leader'):
    st.header('Top 20 Defensive Player')
    playerstats.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')
    df = df[(df['G'] >= 27) & (df['MP'] >=12)]
    # df = df.sort_values(by=['STL%','BLK%','DRB%'],ascending=[False,False,False])
    df = df.sort_values(by=['DRtg'],ascending=[True])
    st.dataframe(df.head(20))
#Visualization
st.subheader('Data Visualization')
playerstats.to_csv('output.csv',index=False)
df = pd.read_csv('output.csv')
team = sorted(df['Tm'].unique())

#lineplot
st.header('NBA gameplay trend')
# current_year = datetime.datetime.now().year
# arr_3PA = []
# arr_2PA = []
# years = []
# def load_line(year):
#     url = 'https://www.basketball-reference.com/leagues/NBA_' \
#         +str(year) + '_per_game.html'
#     html = pd.read_html(url, header = 0)
#     df_per = html[0]
#     data = df_per.drop(df_per[df_per.Age == 'Age'].index)  # Deletes repeating headers in content
#     data = data.fillna(0) 
#     data = data[data['Tm'] != 'TOT']
#     data['3PA'] = data['3PA'].astype(float)
#     data['2PA'] = data['2PA'].astype(float)
#     line_sc = data.groupby('Tm')[['3PA','2PA']].sum()
#     item_3PA = line_sc['3PA'].mean()
#     item_2PA = line_sc['2PA'].mean()
#     return item_3PA,item_2PA

# for year in range(1985,current_year+1):
#     item3, item2 = load_line(year)
#     arr_3PA.append(item3)
#     arr_2PA.append(item2)
#     years.append(year)

data = pd.read_csv('trend.csv')
with sns.axes_style("white"):
        f_line, ax_line = plt.subplots(figsize=(7, 5))
        ax_line = sns.lineplot(x='Year',y='2PA',data = data,color='#0d7c7e',label='2 Point')
        ax_line = sns.lineplot(x='Year',y='3PA',data = data,color='#f08080',label='3 Point')
        ax_line.set_xlabel('Year')
        ax_line.set_ylabel('Attempts Per Game by a team')
        plt.legend()
st.pyplot(f_line)

#Kdeplot
st.header('Distribution of all stats')
stats = list(df.columns)
selected_stats = st.selectbox('Stats',stats)
with sns.axes_style("white"):
        try:
            f_kde, ax_kde = plt.subplots(figsize=(7, 5))
            mean = round(df[selected_stats].mean(),2)
            std = round(df[selected_stats].std(),2)
            ax_kde = sns.kdeplot(df[selected_stats],shade=True,color='red',label=f'mean = {mean}\nstd = {std}')
            ax_kde.legend()
        except:
            plt.title('UNAVAILABLE')
st.pyplot(f_kde)

#Histplot
def histplot_team(Tm):
    df_plot = df[df['Tm'] == Tm]
    with sns.axes_style("white"):
        mean = round(df_plot['PTS'].mean(),2)
        std = round(df_plot['PTS'].std(),2)
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.histplot(x=df_plot['PTS'],bins=10,color='#0d7c7e',kde=True,label=f'mean = {mean}\nstd = {std}')
        ax.set_xlabel('Point Per Game')
        ax.set_ylabel('Player')
        ax.legend()
    st.pyplot(f)
    
st.header('Point Per Game Distribution Each Team')
selected_team_hist = st.selectbox('Team',team,key='hist')
histplot_team(selected_team_hist)

#Barplot
def barplot_team(Tm):
    df_plot = df[df['Tm'] == Tm]
    df_plot['FTS'] = df_plot['FT']
    df_plot['2PS'] = df_plot['2P'] * 2 
    df_plot['3PS'] = df_plot['3P'] * 3 

    with sns.axes_style('white'):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = plt.barh(df_plot['Player'],df_plot['FTS'],label='Free Throw',color='#daa520')
        ax = plt.barh(df_plot['Player'],df_plot['2PS'],left=df_plot['FTS'],label='2 Point',color='#0d7c7e')
        ax = plt.barh(df_plot['Player'],df_plot['3PS'],left=df_plot['2PS']+df_plot['FTS'],label='3 Point',color='#f08080')
        plt.xlabel('Point Per Game')
        plt.legend()
    st.pyplot(f)
    point = df_plot[df_plot['PTS'] >= 15]['PTS']
    star = df_plot[df_plot['PTS'] >= 15]['Player']
    return star,len(star),point

st.header('Point Per Game - Deep Analysis')
selected_team_bar = st.selectbox('Team',team,key='bar')
star,amount,point = barplot_team(selected_team_bar)
if amount > 1:
    st.markdown(f'**=> There are {amount} players who scores more than 15 PPG.**')
else:
    st.markdown(f'**=> There are {amount} player who scores more than 15 PPG.**')

for i in range(amount):
    st.markdown(f'* {star.values[i]} : {point.values[i]} point per game.')

#Pie chart
f_pie,ax_pie = plt.subplots(figsize=(7, 5))
df_pie = df[df['Tm'] == selected_team_bar]
sc_pie = {'3 Point Attempts':df_pie['3PA'].sum(),'2 Point Attempts':df_pie['2PA'].sum()}
ax_pie = plt.pie(x=sc_pie.values(), autopct="%.1f%%", explode=[0.05]*2, labels=sc_pie.keys(), pctdistance=0.5,colors=['#f08080','#0d7c7e'])
plt.title('Gameplay Strategy',size=14)
st.pyplot(f_pie)

st.header('Others Analysis')
#Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f_heat, ax_heat = plt.subplots(figsize=(7, 5))
        ax_heat = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f_heat)

#Boxplot
if st.button('PPG by Pos'):
    st.header('Point Per Game by Position')
    with sns.axes_style("white"):
        f_box, ax_box = plt.subplots(figsize=(7, 5))
        ax_box = sns.boxplot(x='Pos',y='PTS',data=df)
        ax_box.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
        ax_box.set_xlabel('Position')
        ax_box.set_ylabel('Point Per Game')
    st.pyplot(f_box)

if st.button('BLK% by Pos'):
    st.header('Block PCT by Position')
    with sns.axes_style("white"):
        f_box2, ax_box2 = plt.subplots(figsize=(7, 5))
        ax_box2 = sns.boxplot(x='Pos',y='BLK%',data=df)
        ax_box2.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
        ax_box2.set_xlabel('Position')
        ax_box2.set_ylabel('Block PCT')
    st.pyplot(f_box2) 

if st.button('STL% by Pos'):
    st.header('Steal PCT by Position')
    with sns.axes_style("white"):
        f_box3, ax_box3 = plt.subplots(figsize=(7, 5))
        ax_box3 = sns.boxplot(x='Pos',y='STL%',data=df)
        ax_box3.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
        ax_box3.set_xlabel('Position')
        ax_box3.set_ylabel('Steal PCT')
    st.pyplot(f_box3) 

if st.button('AST% by Pos'):
    st.header('Assist PCT by Position')
    with sns.axes_style("white"):
        f_box4, ax_box4 = plt.subplots(figsize=(7, 5))
        ax_box4 = sns.boxplot(x='Pos',y='AST%',data=df)
        ax_box4.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
        ax_box4.set_xlabel('Position')
        ax_box4.set_ylabel('Assist PCT')
    st.pyplot(f_box4) 

if st.button('TRB% by Pos'):
    st.header('Rebound PCT by Position')
    with sns.axes_style("white"):
        f_box5, ax_box5 = plt.subplots(figsize=(7, 5))
        ax_box5 = sns.boxplot(x='Pos',y='TRB%',data=df)
        ax_box5.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
        ax_box5.set_xlabel('Position')
        ax_box5.set_ylabel('Rebound PCT')
    st.pyplot(f_box5) 

if selected_year > 1990:
    if st.button('Salary by Pos'):
        st.header('Salary by Position')
        with sns.axes_style("white"):
            f_box6, ax_box6 = plt.subplots(figsize=(7, 5))
            ax_box6 = sns.boxplot(x='Pos',y='Salary ($)',data=df)
            ax_box6.set_xticklabels(labels=list(df['Pos'].unique()),rotation=30)
            ax_box6.set_xlabel('Position')
            ax_box6.set_ylabel('Salary ($)')
        st.pyplot(f_box6) 

#Regression plot
if st.button('ORB vs DRB'):
    st.header('Offensive versus Defensive Rebound')
    with sns.axes_style("white"):
        f_reg, ax_reg = plt.subplots(figsize=(7, 5))
        ax_reg = sns.regplot(x='ORB',y='DRB',data=df,marker='*',color='#0d7c7e')
        ax_reg.set_xlabel('Offensive Rebound')
        ax_reg.set_ylabel('Defensive Rebound')
    st.pyplot(f_reg)

if st.button('WS,BPM,VORP'):
    st.header('Winshare versus Box Plus Minus')
    with sns.axes_style("white"):
        f_reg1, ax_reg1 = plt.subplots(figsize=(7, 5))
        ax_reg1 = sns.regplot(x='WS',y='BPM',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg1)

    st.header('Winshare versus Value Over Replacement Player')
    with sns.axes_style("white"):
        f_reg2, ax_reg2 = plt.subplots(figsize=(7, 5))
        ax_reg2 = sns.regplot(x='WS',y='VORP',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg2)

    st.header('Box Plus Minus versus Value Over Replacement Player')
    with sns.axes_style("white"):
        f_reg3, ax_reg3 = plt.subplots(figsize=(7, 5))
        ax_reg3 = sns.regplot(x='BPM',y='VORP',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg3)

if st.button('PPG vs WS,BPM,VORP'):
    st.header('Point Per Game versus Winshare')
    with sns.axes_style("white"):
        f_reg4, ax_reg4 = plt.subplots(figsize=(7, 5))
        ax_reg4 = sns.regplot(x='PTS',y='WS',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg4)

    st.header('Point Per Game versus Box Plus Minus')
    with sns.axes_style("white"):
        f_reg5, ax_reg5 = plt.subplots(figsize=(7, 5))
        ax_reg5 = sns.regplot(x='PTS',y='BPM',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg5)

    st.header('Point Per Game versus Value Over Replacement Player')
    with sns.axes_style("white"):
        f_reg6, ax_reg6 = plt.subplots(figsize=(7, 5))
        ax_reg6 = sns.regplot(x='PTS',y='VORP',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg6)

if st.button('STL%, DRB%, BLK%'):
    st.header('Steal PCT versus Defensive Rebound PCT')
    with sns.axes_style("white"):
        f_reg7, ax_reg7 = plt.subplots(figsize=(7, 5))
        ax_reg7 = sns.regplot(x='STL%',y='DRB%',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg7)

    st.header('Steal PCT versus Block PCT')
    with sns.axes_style("white"):
        f_reg8, ax_reg8 = plt.subplots(figsize=(7, 5))
        ax_reg8 = sns.regplot(x='STL%',y='BLK%',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg8)

    st.header('Defensive Rebound PCT versus Block PCT')
    with sns.axes_style("white"):
        f_reg9, ax_reg9 = plt.subplots(figsize=(7, 5))
        ax_reg9 = sns.regplot(x='DRB%',y='BLK%',data=df,marker='*',color='#0d7c7e')
    st.pyplot(f_reg9)

for i in range(50):
    st.write('')
cl = Image.open('cl.jpg')
st.image(cl,caption='thanks for visiting!')

