import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import numpy as np 
import base64
import pickle

st.title("Simulador - Sobrevivência do COVID-19")
st.image("logo_covid.jpg",width = 550)
nav = st.sidebar.radio("Navegação",["Home","Predição","Gráficos"])

if nav == "Home":
    st.markdown("O objetivo principal desta ferramenta é realizar predições sobre a chance de um paciente sobreviver considerando as suas comorbidades caso seja contaminado pelo COVID 19")
    st.title("Projeto Integrador COVID 19 - 2021")
    st.title("Curso Data Science - Digital House")
    
if nav == "Predição":
    
    with st.sidebar:
        st.sidebar.header('Entrada de dados do usuário:')
        st.sidebar.markdown("""[Exemplo de arquivo CSV](https://raw.githubusercontent.com/hrsousa/projetointegrador_covid19/main/exemplo_dados.csv)""")
        database = st.radio('Seleção da fonte dos dados de entrada (X):',('Manual', 'CSV'))        
        if database == 'CSV':
            st.info('Upload do CSV')
            file = st.file_uploader('Selecione o arquivo CSV contendo as colunas acima descritas',type='csv')
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)				    
            if file:
                Xtest = pd.read_csv(file)
                mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
                ypred = mdl_lgbm.predict(Xtest)
        else:
            X1 = st.number_input('1. Idade do paciente de 0 a 130 anos',0,130)
            X2 = st.sidebar.selectbox('2. Qual o sexo do paciente? (0 - Feminino | 1 - Masculino)',[0,1])
            X3 = st.sidebar.selectbox('3. Cor ou raça declarada pelo paciente? (0 - Outras | 1 - Parda)',[0,1])
            X4 = st.sidebar.selectbox('4. Paciente apresentou febre? (0 - Sem febre | 1 - Com febre)',[0,1])
            X5 = st.sidebar.selectbox('5. Paciente apresentou dispneia? (0 - Com dispneia | 1 - Sem dispneia)',[0,1]) 
            X6 = st.sidebar.selectbox('6. Paciente apresentou saturação O2 menor que 95%? (0 - Saturação Normal | 1 - Saturação Baixa)',[0,1])	
            X7 = st.sidebar.selectbox('7. Paciente foi internado em UTI?(0 - Não Internado na UTI | 1 - Internado na UTI)',[0,1])
            X8 = st.sidebar.selectbox('8. Paciente fez uso de suporte ventilatório (0 - Não Teve Suporte Invasivo | 1 - Teve Suporte Invasivo)?',[0,1])
            X9 = st.sidebar.selectbox('9. Paciente fez uso de suporte ventilatório (0 - Não Teve Sup. Não Invas. | 1 - Teve Sup. Não Invas.)?',[0,1]) 
            X10 = st.sidebar.selectbox('10. Resultado teste de RT-PCR/outro método por Biol. Molecular (0 - Outras | 1 - PCR não Detectavel)',[0,1]) 
            X11 = st.sidebar.selectbox('11. Paciente apresentou tosse? (0 - Sem  tosse | 1 - Com tosse)',[0,1])
            X12 = st.sidebar.selectbox('12. Paciente apresentou desconforto respiratório? (0 - Sem desconforto | 1 - Com desconforto)',[0,1]) 
            X13 = st.sidebar.selectbox('13. Paciente possui outro(s) fator(es) de risco? (0 - Não Possui Fat. de Risco | 1 - Possui Fat. de Risco)',[0,1])

            Xtest = pd.DataFrame({'IDADE_ANOS': [X1], 'CS_SEXO_M': [X2], 'CS_RACA_4.0': [X3], 'FEBRE_1.0': [X4], 
                                      'DISPNEIA_1.0': [X5], 'SATURACAO_1.0': [X6], 'UTI_1.0': [X7], 'SUPORT_VEN_1.0': [X8], 
                                      'SUPORT_VEN_2.0': [X9], 'PCR_RESUL_2.0': [X10], 'TOSSE_1.0': [X11], 
                                      'DESC_RESP_1.0': [X12], 'FATOR_RISC_2': [X13]})
                                  
            
            mdl_lgbm = pickle.load(open('pickle_mdl_lregression_select.pkl', 'rb'))
            ypred = mdl_lgbm.predict(Xtest)
                                     
##################################################################################################################

    if database == 'Manual':
        with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
        with st.expander('Visualizar Predição', expanded = True):
                if ypred==2:
                    st.error(ypred[0])
                else:
                    st.success(ypred[0])
                    
        if st.button('Baixar arquivo csv'):
            df_download = Xtest.copy()
            df_download['Response_pred'] = ypred
            st.dataframe(df_download)
            csv = df_download.to_csv(sep=',',decimal=',',index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            

    else: #database == 'CSV'
        if file:
            with st.expander('Visualizar Dados de Entrada', expanded = False):
                st.dataframe(Xtest)
            with st.expander('Visualizar Predições', expanded = False):
                st.dataframe(ypred)            
            
            if st.button('Baixar arquivo csv'):
                df_download = Xtest.copy()
                df_download['Response_pred'] = ypred
                st.write(df_download.shape)
                st.dataframe(df_download)
                csv = df_download.to_csv(sep=',',decimal=',',index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)    

if nav == "Gráficos":
    st.header("Plotagem de gráficos da pandemia")
    
    st.sidebar.markdown("""[Base de dados CSV (DATASUS)](https://raw.githubusercontent.com/hrsousa/projetointegrador_covid19/main/worldometer_data.csv)""")  
    
    import logging
    st.title("Covid 19 - Maiores Municipios - 24/09/2021")
    DATA_URL =  "worldometer_data.csv"

    @st.cache
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data

    #
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text("Done! (using st.cache)")
    #
    show_data_bol = st.sidebar.checkbox("Show Raw data")
    if show_data_bol:
        st.subheader('Raw data')
        st.write(data)
    #
    st.subheader('Total de: Casos, Óbitos, Vacinados 1 dose, Vacinados 2 doses')
    #
    columns = ["Casos", "Obitos", "Vacinados_1_Dose", "Vacinados_2_doses"]
    for x in columns:	
        fig = px.treemap(data.iloc[0:19],path=['Municipios'],values=x,title="treemap Covid  %s" %x)
        st.plotly_chart(fig)
