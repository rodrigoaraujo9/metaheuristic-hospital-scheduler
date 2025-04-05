import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from parser import parse_instance_file
from main import get_scheduler
from utils import analyze_solution  # new analysis function

st.set_page_config(page_title="Agendamento Hospitalar", layout="wide")

# Sidebar configuration
st.sidebar.title("Configurações")
algoritmo = st.sidebar.selectbox("Escolha o algoritmo", ["sa", "tabu", "ga"])

instancias_path = "./data/instances"
arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]

usar_todas = st.sidebar.checkbox("Usar todas as instâncias", value=False)
instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Selecionar instâncias", arquivos)

if st.sidebar.button("Executar"):
    resultados = []
    
    for filename in instancias_escolhidas:
        instance_name = os.path.splitext(filename)[0]
        with st.expander(f"Instância: {instance_name}", expanded=True):
            filepath = os.path.join(instancias_path, filename)
            dados = parse_instance_file(filepath)
            scheduler = get_scheduler(algoritmo, dados)
            solucao = scheduler.run()
            
            # Use the new analysis function to extract detailed stats
            stats = analyze_solution(dados, solucao)
            
            total_pacientes = stats['total_patients']
            alocados = stats['allocated_patients']
            pct_alocados = (alocados / total_pacientes) * 100
            custo = getattr(scheduler, "final_cost", None)
            tempo = getattr(scheduler, "runtime", "N/A")
            iteracoes = getattr(scheduler, "iterations", getattr(scheduler, "generations", "N/A"))
            
            st.markdown("### Estatísticas Gerais")
            st.write({
                "Iterações/Gerações": iteracoes,
                "Tempo (s)": round(tempo, 2),
                "% Alocados": round(pct_alocados, 2),
                "Custo Final": custo
            })
            
            st.markdown("### Análise da Solução")
            st.write("**Cost Breakdown:**", stats['cost_breakdown'])
            st.write("**OT Utilization:**", stats['ot_utilization'])
            st.write("**Ward Occupancy:**")
            st.write(stats['ward_occupancy'])
            
            # Plot the cost history and, if SA, the temperature history
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(scheduler.cost_history, label="Custo")
                ax1.set_title("Evolução do Custo")
                ax1.set_xlabel("Iteração" if algoritmo != "ga" else "Geração")
                ax1.set_ylabel("Custo")
                ax1.grid()
                ax1.legend()
                st.pyplot(fig1)
            
            if algoritmo == "sa":
                with col2:
                    fig2, ax2 = plt.subplots()
                    ax2.plot(scheduler.temperature_history, label="Temperatura", color="orange")
                    ax2.set_title("Evolução da Temperatura")
                    ax2.set_xlabel("Iteração")
                    ax2.set_ylabel("Temperatura")
                    ax2.grid()
                    ax2.legend()
                    st.pyplot(fig2)
            
            # Additional plot: Number of surgeries per day
            fig3, ax3 = plt.subplots()
            ax3.bar(range(dados['days']), stats['surgery_per_day'])
            ax3.set_title("Cirurgias por Dia")
            ax3.set_xlabel("Dia")
            ax3.set_ylabel("Número de Cirurgias")
            st.pyplot(fig3)
            
            resultados.append({
                "Instância": filename,
                "Iterações/Gerações": iteracoes,
                "Tempo (s)": round(tempo, 2),
                "% Alocados": round(pct_alocados, 2),
                "Custo Final": custo
            })
    
    st.markdown("## Estatísticas Resumidas")
    df_resultados = pd.DataFrame(resultados)
    st.dataframe(df_resultados)