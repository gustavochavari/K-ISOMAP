import sys
import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore')

def main():
    # Choose .json results
    with open(r'C:\Users\Gustavo\Documents\Mestrado\artigos\K-ISOMAP\first_set_experiments.json', 'r') as file:
        results = json.load(file)

    rows = []

    for dataset_name, methods in results.items():
        for method, values in methods.items():
            if values and len(values[0]) == 6:  # Verifica se há valores e se têm pelo menos 6 elementos
                rows.append([dataset_name, method, values[0][0], values[0][1], values[0][2], values[0][3], values[0][4], values[0][5]])
            else:
                rows.append([dataset_name, method, values[0][0], values[1][0], values[2][0], values[3][0], values[4][0], values[5][0]])

    df = pd.DataFrame(rows, columns=['Dataset', 'Method', 'RI', 'CH', 'FM', 'VS', 'SS', 'DB'])


    metrics = ["RI", "CH", "FM", "VS", "SS", "DB"]

    methods = ['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']

    # Retirando resultados normalizados, pegando apenas os valores brutos
    mask = ~df['Dataset'].str.contains('_norm')
    #mask_norm = df['Dataset'].str.contains('norm')

    # Dataset with actual values
    df_filtered = df[mask]

    df = df_filtered

    statistics = []

    # Iterando sobre todas as combinações de métricas e métodos
    for metric in metrics:
        if '_norm' not in df['Dataset'].unique():
            kiso_data = df[df['Method'] == 'KISOMAP'][metric]
            raw_data = df[df['Method'] == 'RAW'][metric]
        for method in methods:
            if method != 'KISOMAP' and method != "RAW":  # Comparar com KISOMAP
                other_data = df[df['Method'] == method][metric]
                if len(kiso_data) == len(other_data):  # Certificar-se de que temos o mesmo número de amostras
                    # Realizando os testes
                    stat, p_value = friedmanchisquare(raw_data, kiso_data, other_data)
                    p_values_nemenyi = sp.posthoc_nemenyi_friedman(np.array([raw_data, kiso_data, other_data]).T)

                    # Armazenando os resultados
                    statistics.append({
                        'Metric': metric,
                        'Method': method,
                        'Stat': stat,
                        'P-Value Friedman': p_value,
                        # Esse aqui é o valor do teste do KISOMAP vs. (method)
                        'P-Value Nemenyi': p_values_nemenyi[0][2]
                    })

                else:
                    print(f'Número de amostras não corresponde entre KISOMAP e {method} para a métrica {metric}')

    results_df = pd.DataFrame(statistics)

    results_df['Test Friedman'] = results_df['P-Value Friedman'].apply(lambda x: 1 if x < 0.05 else 0)
    results_df['Stat'] = results_df['Stat'].apply(lambda x: f"{x:.3f}")
    results_df['Test Nemenyi'] = results_df['P-Value Nemenyi'].apply(lambda x: 1 if x < 0.05 else 0)
    results_df['P-Value Friedman'] = results_df['P-Value Friedman'].apply(lambda x: f"{x:.3f}")
    results_df['P-Value Nemenyi'] = results_df['P-Value Nemenyi'].apply(lambda x: f"{x:.3f}")

    results_df = results_df.drop('Stat',axis=1)

    results_df[['Method','Metric','P-Value Friedman','P-Value Nemenyi','Test Friedman','Test Nemenyi']].sort_values(['Method','Metric'])

    ###### Exportar tabela em latex
    latex_code = results_df[['Method','Metric','P-Value Friedman','P-Value Nemenyi','Test Friedman','Test Nemenyi']].sort_values(['Method','Metric']).to_latex(index=False)

    with open("1st_battery_results.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)

    df_melted = pd.melt(df_filtered, id_vars=['Dataset', 'Method'], 
                    value_vars=['RI',
                                'CH',
                                'FM',
                                'VS', 
                                'SS',
                                'DB'],
                    var_name='metric', value_name='value')

    # Pivotando para criar uma tabela com subcolunas para cada métrica e método
    df_pivot = df_melted.pivot_table(index='Dataset', 
                                     columns=['metric', 'Method'], 
                                     values='value')

    df_pivot = df_pivot.swaplevel(i=0, j=1, axis=1)

    df_pivot = df_pivot.sort_index(axis=1)


    ## Plot Boxplot
    df_gmm_ri = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='RI')
    df_gmm_ri.reset_index(inplace=True) 
    latex_code = df_gmm_ri.round(3).to_latex(index=False)
    with open("1st_battery_results_RI.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)

    df_gmm_ch = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='CH')
    df_gmm_ch.reset_index(inplace=True)  
    latex_code = df_gmm_ch.round(3).to_latex(index=False)
    with open("1st_battery_results_CH.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)

    df_gmm_fm = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='FM')
    df_gmm_fm.reset_index(inplace=True)
    latex_code = df_gmm_fm.round(3).to_latex(index=False)
    with open("1st_battery_results_FM.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)  

    df_gmm_vs = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='VS')
    df_gmm_vs.reset_index(inplace=True)
    latex_code = df_gmm_vs.round(3).to_latex(index=False)
    with open("1st_battery_results_VS.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)  

    df_gmm_ss = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='SS')
    df_gmm_ss.reset_index(inplace=True)  
    latex_code = df_gmm_ss.round(3).to_latex(index=False)
    with open("1st_battery_results_SS.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)


    df_gmm_db = df_pivot[['ISOMAP','KISOMAP','KPCA','LLE','SE','TSNE','UMAP']].filter(like='DB')
    df_gmm_db.reset_index(inplace=True)  
    latex_code = df_gmm_db.round(3).to_latex(index=False)
    with open("1st_battery_results_DB.txt", "w", encoding="utf-8") as file:
        file.write(latex_code)

    results_data = {
        'RI': [df_gmm_ri[['Dataset','ISOMAP']]['ISOMAP']['RI'],
                 df_gmm_ri[['Dataset','KISOMAP']]['KISOMAP']['RI'],
                 df_gmm_ri[['Dataset','KPCA']]['KPCA']['RI'],
                 df_gmm_ri[['Dataset','LLE']]['LLE']['RI'],
                 df_gmm_ri[['Dataset','SE']]['SE']['RI'],
                 df_gmm_ri[['Dataset','TSNE']]['TSNE']['RI'],
                 df_gmm_ri[['Dataset','UMAP']]['UMAP']['RI']],
        'CH': [df_gmm_ch[['Dataset','ISOMAP']]['ISOMAP']['CH'],
                 df_gmm_ch[['Dataset','KISOMAP']]['KISOMAP']['CH'],
                 df_gmm_ch[['Dataset','KPCA']]['KPCA']['CH'],
                 df_gmm_ch[['Dataset','LLE']]['LLE']['CH'],
                 df_gmm_ch[['Dataset','SE']]['SE']['CH'],
                 df_gmm_ch[['Dataset','TSNE']]['TSNE']['CH'],
                 df_gmm_ch[['Dataset','UMAP']]['UMAP']['CH']],
        'FM': [df_gmm_fm[['Dataset','ISOMAP']]['ISOMAP']['FM'],
                 df_gmm_fm[['Dataset','KISOMAP']]['KISOMAP']['FM'],
                 df_gmm_fm[['Dataset','KPCA']]['KPCA']['FM'],
                 df_gmm_fm[['Dataset','LLE']]['LLE']['FM'],
                 df_gmm_fm[['Dataset','SE']]['SE']['FM'],
                 df_gmm_fm[['Dataset','TSNE']]['TSNE']['FM'],
                 df_gmm_fm[['Dataset','UMAP']]['UMAP']['FM']],
        'VS': [df_gmm_vs[['Dataset','ISOMAP']]['ISOMAP']['VS'],
                 df_gmm_vs[['Dataset','KISOMAP']]['KISOMAP']['VS'],
                 df_gmm_vs[['Dataset','KPCA']]['KPCA']['VS'],
                 df_gmm_vs[['Dataset','LLE']]['LLE']['VS'],
                 df_gmm_vs[['Dataset','SE']]['SE']['VS'],
                 df_gmm_vs[['Dataset','TSNE']]['TSNE']['VS'],
                 df_gmm_vs[['Dataset','UMAP']]['UMAP']['VS']],
        'SS': [df_gmm_ss[['Dataset','ISOMAP']]['ISOMAP']['SS'],
                 df_gmm_ss[['Dataset','KISOMAP']]['KISOMAP']['SS'],
                 df_gmm_ss[['Dataset','KPCA']]['KPCA']['SS'],
                 df_gmm_ss[['Dataset','LLE']]['LLE']['SS'],
                 df_gmm_ss[['Dataset','SE']]['SE']['SS'],
                 df_gmm_ss[['Dataset','TSNE']]['TSNE']['SS'],
                 df_gmm_ss[['Dataset','UMAP']]['UMAP']['SS']],
        'DB': [df_gmm_db[['Dataset','ISOMAP']]['ISOMAP']['DB'],
                 df_gmm_db[['Dataset','KISOMAP']]['KISOMAP']['DB'],
                 df_gmm_db[['Dataset','KPCA']]['KPCA']['DB'],
                 df_gmm_db[['Dataset','LLE']]['LLE']['DB'],
                 df_gmm_db[['Dataset','SE']]['SE']['DB'],
                 df_gmm_db[['Dataset','TSNE']]['TSNE']['DB'],
                 df_gmm_db[['Dataset','UMAP']]['UMAP']['DB']]
    } 

    # Criando subplots para as métricas
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)

    # Definindo as cores para os métodos KISOMAP e ISOMAP
    method_colors = {
    "ISOMAP": "#F8B717",   # Laranja
    "KISOMAP": "#4A9FFF"  # Azul
    }

    # Iterando sobre as métricas e eixos para criar os gráficos
    for ax, metric in zip(axes.flat, metrics):
        data = results_data[metric]
        # Criando o boxplot com patch_artist=True para permitir coloração
        box = ax.boxplot(data, labels=methods, patch_artist=True)

        # Colorindo apenas os métodos KISOMAP e ISOMAP
        for patch, label in zip(box['boxes'], methods):
            if label in method_colors:
                patch.set_facecolor(method_colors[label])  # Define a cor do método
            else:
                patch.set_facecolor("white")  # Outros métodos sem cor

        # Configurações do gráfico
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if metric == 'CH':
            ax.set_ylim(-100, 5000)  # Ajuste para o intervalo desejado

    # Salvando os gráficos
    plt.savefig('1st_battery_boxplots.jpeg', format='jpeg', dpi=300)
    plt.close()


    ## Plot Lineplot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.style.use('seaborn')

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    # Create subplots for each metric
    for idx, metric in enumerate(metrics):
        # Sort dataframe to put KISOMAP last for this metric
        df_sorted = df.copy()
        df_sorted['plot_order'] = df_sorted['Method'].map(lambda x: 2 if x == 'KISOMAP' else 1 if x == 'ISOMAP' else 0)
        df_sorted = df_sorted.sort_values('plot_order')

        # Define custom color palette
        custom_palette = {method: '#023EFF' if method == 'KISOMAP' else '#A9A9A9' if method != 'ISOMAP' else '#FF7C00'
                         for method in df_sorted['Method'].unique()}

        # Create the line plot for this metric
        sns.lineplot(data=df_sorted, x='Dataset', y=metric, hue='Method',
                    palette=custom_palette, linewidth=2.5, ax=axes_flat[idx])

        # Customize each subplot
        #axes_flat[idx].set_xlabel('Datasets')
        axes_flat[idx].set_ylabel(f'{metric} Values')
        if idx == 1:
            axes_flat[idx].set_ylim(-100,5000)
        axes_flat[idx].set_title(f'{metric} Metric')
        axes_flat[idx].tick_params(axis='x', rotation=45, labelbottom=False)
        # Move legend outside of plot
        
        if idx == 2:
            axes_flat[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes_flat[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes_flat[idx].get_legend().remove()


    plt.savefig('1st_battery_lineplots.jpeg',format='jpeg',dpi=300)
    plt.close()

if __name__ == '__main__':
    sys.exit(main())