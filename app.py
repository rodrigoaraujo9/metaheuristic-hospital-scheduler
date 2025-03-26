# app.py
import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from parser import parse_instance_file
from main import get_scheduler
import shutil

st.set_page_config(page_title="Agendamento Hospitalar", layout="wide")

# Sidebar
st.sidebar.title("Configurações")
algoritmo = st.sidebar.selectbox("Escolha o algoritmo", ["sa", "tabu", "ga"])

instancias_path = "./data/instances"
arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]

usar_todas = st.sidebar.checkbox("Usar todas as instâncias", value=True)
instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Selecionar instâncias", arquivos)

if st.sidebar.button("Executar"):
    resultados = []
    graficos = {}

    for filename in instancias_escolhidas:
        st.subheader(f"Instância: {filename}")
        filepath = os.path.join(instancias_path, filename)
        dados = parse_instance_file(filepath)
        scheduler = get_scheduler(algoritmo, dados)
        solucao = scheduler.run()

        # Estatísticas
        total_pacientes = len(dados['patients'])
        alocados = sum(1 for p in solucao if solucao[p]['ward'] is not None)
        pct_alocados = (alocados / total_pacientes) * 100
        custo = getattr(scheduler, "final_cost", None)
        tempo = getattr(scheduler, "runtime", "N/A")
        iteracoes = getattr(scheduler, "iterations", getattr(scheduler, "generations", "N/A"))

        resultados.append({
            "Instância": filename,
            "Iterações/Gerações": iteracoes,
            "Tempo (s)": round(tempo, 2),
            "% Alocados": round(pct_alocados, 2),
            "Custo Final": custo
        })

        # Plot gráfico de custo
        fig1, ax1 = plt.subplots()
        ax1.plot(scheduler.cost_history, label="Custo")
        ax1.set_title("Evolução do Custo")
        ax1.set_xlabel("Iteração" if algoritmo != "ga" else "Geração")
        ax1.set_ylabel("Custo")
        ax1.grid()
        ax1.legend()
        st.pyplot(fig1)

        # Se for SA, plotar temperatura
        if algoritmo == "sa":
            fig2, ax2 = plt.subplots()
            ax2.plot(scheduler.temperature_history, label="Temperatura", color="orange")
            ax2.set_title("Evolução da Temperatura")
            ax2.set_xlabel("Iteração")
            ax2.set_ylabel("Temperatura")
            ax2.grid()
            ax2.legend()
            st.pyplot(fig2)

    # Exibir resultados finais
    df_resultados = pd.DataFrame(resultados)
    st.markdown("## Estatísticas Resumidas")
    st.dataframe(df_resultados)
