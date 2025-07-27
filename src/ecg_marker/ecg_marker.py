# ECG Marker - Ferramenta Interativa de Marcação de Eventos em Sinais de ECG

# Este programa permite a visualização e anotação interativa de traçados de ECG a partir de arquivos
# de entrada (dados brutos ou previamente anotados). 

# Desenvolvido por: Thaís de Jesus Soares

# Bibliotecas
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.widgets as widgets
import numpy as np
import argparse
import os
import neurokit2 as nk

# Lista com todos os nomes de canais (eletrodos) esperados nos arquivos de entrada
head_file = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'HISp', 'HISd', 'VD p', 'VD 78', 'VD 56', 'VD 34', 'VD d']
# Subconjunto de canais que vão ser processados
head      = ['VD d', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def read_file(filename):
    
    # Lê um arquivo CSV contendo sinais de eletrodos e retorna os dados organizados.

    # Parâmetros:
    #     filename (str): Caminho para o arquivo CSV a ser lido.

    # Retorno:
    #     infos (dict): Dicionário contendo os valores dos eletrodos especificados em `head`.
    #                   Estrutura: { 'nome_do_eletrodo': {'values': [valores]} }
    #     num_lines (int): Número de linhas de dados (amostras) no arquivo.

    with open(filename, 'r') as f:
        indexes = []
        for i in head:
            indexes.append(head_file.index(i))

        infos = {}
        for i in indexes:
            infos[head_file[i]] = {'values': []}
    
        f.readline()
        
        lines     = f.readlines()
        num_lines = len(lines)
        for line in lines:
            line = line.split(sep=',')
            
            for i in indexes:
                infos[head_file[i]]['values'].append(float(line[i]))
    return infos, num_lines

def read_dir(input_dir):

    # Lê todos os arquivos CSV de um diretório contendo sinais de eletrodos e 
    # agrega os dados em um único dicionário.

    # Parâmetros:
    #     input_dir (str): Caminho para o diretório contendo os arquivos CSV.

    # Retorno:
    #     infos (dict): Dicionário contendo os valores dos eletrodos especificados em `head`.
    #                   Os dados são concatenados de todos os arquivos encontrados.
    #                   Estrutura: { 'nome_do_eletrodo': {'values': [valores]} }
    #     num_lines (int): Número total de linhas de dados (amostras) somando todos os arquivos.

    indexes = []
    for i in head:
        indexes.append(head_file.index(i))

    infos = {}
    for i in indexes:
        infos[head_file[i]] = {'values': []}

    num_lines = 0

    for filename in os.listdir(input_dir):
        file = os.path.join(input_dir, filename)
        if os.path.isfile(file) == False:
            continue

        with open(file, 'r') as f:
        
            f.readline()
            
            lines     = f.readlines()
            num_lines += len(lines)
            for line in lines:
                line = line.split(sep=',')
                
                for i in indexes:
                    infos[head_file[i]]['values'].append(float(line[i]))
                    
    return infos, num_lines

def update(val):

    # Atualiza a visualização do gráfico com base na posição da barra de rolagem.

    # Parâmetros:
    #     val (float): Novo valor da barra de rolagem (posição inicial da janela de visualização).

    start_index = int(scrollbar.val)
        
    if (start_index + xlim < num_lines):
        ax.set_xlim(start_index, start_index + xlim)
    else:
        ax.set_xlim(start_index - xlim, start_index)

    fig.canvas.draw_idle()

def on_enter(event):

    # Altera o tamanho da janela de visualização (xlim) com base na entrada do usuário 
    # na caixa de texto, e atualiza o gráfico.

    # Parâmetros:
    #     event: Evento de pressionar Enter (fornecido automaticamente pelo matplotlib)

    global xlim
    textbox_value = textbox.get()
    try:
        new_xlim = int(textbox_value)
        if new_xlim <= num_lines and new_xlim >= 0:
            xlim = new_xlim
            update(scrollbar.val)
        else:
            print("Invalid Limits.")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

def onclick(event):

    # Lida com cliques do mouse no gráfico para selecionar intervalos temporais no sinal.

    # Parâmetros:
    #     event: Evento do matplotlib associado a um clique do mouse.

    # Funcionalidade:
    #     - Quando o botão esquerdo do mouse (event.button == 1) é clicado sobre o gráfico (event.inaxes == ax):
    #         - Se for o primeiro clique (janela.click_state == 0):
    #             - Remove a linha vertical anterior (se existir) com o label 'vertical_line_1'.
    #             - Adiciona a coordenada x do clique à lista `janela.line_coords`.
    #             - Desenha uma linha vertical vermelha tracejada na posição clicada.
    #             - Atualiza o estado de clique para 1.
    #         - Se for o segundo clique (janela.click_state == 1):
    #             - Remove a linha vertical anterior com o label 'vertical_line_2'.
    #             - Adiciona a nova coordenada x.
    #             - Calcula o intervalo entre os dois cliques (em unidades do eixo x).
    #             - Desenha uma linha vertical azul tracejada na nova posição.
    #             - Atualiza o estado de clique para 2.
    #             - Atualiza uma mensagem na interface (`message_label`) com instruções para classificar o intervalo.

    if event.inaxes == ax and event.button == 1:
        if janela.click_state == 0:
            for line in ax.lines:
                if line.get_label() == 'vertical_line_1':
                    line.remove()
                    break
            janela.line_coords.append(event.xdata)
            janela.click_state = 1
            ax.axvline(event.xdata, color = 'r', linestyle = '--', label = 'vertical_line_1', linewidth = 1)
            fig.canvas.draw()
        elif janela.click_state == 1:
            for line in ax.lines:
                if line.get_label() == 'vertical_line_2':
                    line.remove()
                    break
            janela.line_coords.append(event.xdata)
            janela.click_state = 2
            janela.interval = abs(janela.line_coords[-1] - janela.line_coords[-2])
            ax.axvline(event.xdata, color = 'b', linestyle = '--', label = 'vertical_line_2', linewidth = 1)
            fig.canvas.draw()
            message_label.config(text = "Press 'F' to add Period, 'R' to add QRS, 'T' to add QT, 'E' to add extrasystole, 'A' to add arrhythmia or 'C' to cancel.")

def automatic_period_marking():

    # Realiza a marcação automática de períodos cardíacos (R-peaks, QRS, QT) em sinais eletrocardiográficos (ECG) 
    # de múltiplos eletrodos selecionados.

    # Funcionalidade:
    #     - Lê sinais de eletrodos definidos pelo usuário via `select_electrodes()`.
    #     - (Opcional) Aplica limpeza do sinal com neurokit2 (`nk.ecg_clean`).
    #     - Detecta automaticamente:
    #         - Picos R (R-peaks)
    #         - Inícios e finais de onda R (onsets e offsets)
    #         - Finais de onda T
    #     - Sincroniza as detecções entre diferentes eletrodos (alinhando picos similares no tempo).
    #     - Calcula medianas das detecções para extrair batimentos representativos.
    #     - Calcula parâmetros:
    #         - Frequência cardíaca (intervalo RR)
    #         - Duração do complexo QRS
    #         - Intervalo QT
    #     - Atualiza as tabelas gráficas (`freq_table`, `qrs_table`, `qt_table`) com os dados extraídos.
    #     - Mostra mensagens no rótulo da interface (`message_label`) indicando o progresso.

    signals = select_electrodes()

    message_label.config(text = "Making Automatic Markings...")
    message_label.update_idletasks()

    electrodes_ecg_r_peaks   = []
    electrodes_ecg_r_onsets  = []
    electrodes_ecg_r_offsets = []
    electrodes_ecg_t_offsets = []

    max_len = 0

    for electrode in signals:
        print(f'Electrode: {electrode}')

        signal = electrodes[electrode]['values']

        if (clean_signal):
            signal = nk.ecg_clean(signal)

        _, rpeaks = nk.ecg_peaks(signal)

        ecg_r_peaks = rpeaks['ECG_R_Peaks']

        signal_cwt, waves_cwt = nk.ecg_delineate(signal, rpeaks, method="cwt", show=True)

        ecg_r_onsets  = waves_cwt['ECG_R_Onsets']
        ecg_r_offsets = waves_cwt['ECG_R_Offsets']
        ecg_t_offsets = waves_cwt['ECG_T_Offsets']

        print(f'ECG_R_Peaks: {len(ecg_r_peaks)}, ECG_R_Onsets: {len(ecg_r_onsets)}, ECG_R_Offsets: {len(ecg_r_offsets)}, ECG_T_Offsets: {len(ecg_t_offsets)}')

        if max_len < max(len(ecg_r_peaks), len(ecg_r_onsets), len(ecg_r_offsets), len(ecg_t_offsets)):
            max_len = max(len(ecg_r_peaks), len(ecg_r_onsets), len(ecg_r_offsets), len(ecg_t_offsets))
    
        electrodes_ecg_r_peaks.append(ecg_r_peaks)
        electrodes_ecg_r_onsets.append(ecg_r_onsets)
        electrodes_ecg_r_offsets.append(ecg_r_offsets)
        electrodes_ecg_t_offsets.append(ecg_t_offsets)

    new_electrodes_ecg_r_peaks   = np.zeros((len(signals), max_len))
    new_electrodes_ecg_r_onsets  = np.zeros((len(signals), max_len))
    new_electrodes_ecg_r_offsets = np.zeros((len(signals), max_len))
    new_electrodes_ecg_t_offsets = np.zeros((len(signals), max_len))

    loop = True

    cont = np.zeros(len(signals))

    cont_list = 0

    while (loop):

        valid_values = []

        loop = False

        for i in range(len(signals)):
            if cont[i] < len(electrodes_ecg_r_peaks[i]):
                loop = True
                valid_values.append(electrodes_ecg_r_peaks[i][int(cont[i])])

        if (loop == False):
            break

        # median_value = np.median(valid_values)
        min_value = min(valid_values)

        # for i in range(len(signals)):
        #     if cont[i] < len(electrodes_ecg_r_peaks[i]) and abs(electrodes_ecg_r_peaks[i][int(cont[i])] - min_value) < 300:
        #         new_electrodes_ecg_r_peaks[i][cont_list] = electrodes_ecg_r_peaks[i][int(cont[i])]
        #         new_electrodes_ecg_r_onsets[i][cont_list] = electrodes_ecg_r_onsets[i][int(cont[i])] 
        #         new_electrodes_ecg_r_offsets[i][cont_list] = electrodes_ecg_r_offsets[i][int(cont[i])] 
        #         new_electrodes_ecg_t_offsets[i][cont_list] = electrodes_ecg_t_offsets[i][int(cont[i])] 
        #         cont[i] += 1
        #     else:
        #         new_electrodes_ecg_r_peaks[i][cont_list] = np.nan
        #         new_electrodes_ecg_r_onsets[i][cont_list] = np.nan
        #         new_electrodes_ecg_r_offsets[i][cont_list] = np.nan
        #         new_electrodes_ecg_t_offsets[i][cont_list] = np.nan 

        for i in range(len(signals)):
            if (cont[i] < len(electrodes_ecg_r_peaks[i]) and abs(electrodes_ecg_r_peaks[i][int(cont[i])] - min_value) < 300):
                new_electrodes_ecg_r_peaks[i, cont_list] = electrodes_ecg_r_peaks[i][int(cont[i])]
                new_electrodes_ecg_r_onsets[i, cont_list] = (
                    electrodes_ecg_r_onsets[i][int(cont[i])]
                    if int(cont[i]) < len(electrodes_ecg_r_onsets[i])
                    else np.nan
                )
                new_electrodes_ecg_r_offsets[i, cont_list] = (
                    electrodes_ecg_r_offsets[i][int(cont[i])]
                    if int(cont[i]) < len(electrodes_ecg_r_offsets[i])
                    else np.nan
                )
                new_electrodes_ecg_t_offsets[i, cont_list] = (
                    electrodes_ecg_t_offsets[i][int(cont[i])]
                    if int(cont[i]) < len(electrodes_ecg_t_offsets[i])
                    else np.nan
                )
                cont[i] += 1
            else:
                new_electrodes_ecg_r_peaks[i, cont_list] = np.nan
                new_electrodes_ecg_r_onsets[i, cont_list] = np.nan
                new_electrodes_ecg_r_offsets[i, cont_list] = np.nan
                new_electrodes_ecg_t_offsets[i, cont_list] = np.nan

        cont_list += 1

        if cont_list >= max_len:
            loop = False
            break

    for j in range(1, max_len):
        median_r_peaks = np.nanmedian(new_electrodes_ecg_r_peaks[:, j])
        median_r_peaks_last = np.nanmedian(new_electrodes_ecg_r_peaks[:, j - 1])
        median_r_offsets = np.nanmedian(new_electrodes_ecg_r_offsets[:, j])
        median_r_onsets = np.nanmedian(new_electrodes_ecg_r_onsets[:, j])
        median_t_offsets = np.nanmedian(new_electrodes_ecg_t_offsets[:,j])

        if (np.isnan(median_r_peaks) or np.isnan(median_r_peaks_last) or np.isnan(median_r_onsets) or np.isnan(median_r_offsets) or np.isnan(median_t_offsets)):
            continue

        if (median_r_peaks - median_r_peaks_last > 1.1 * 600 or median_r_peaks - median_r_peaks_last < 0.7 * 200):
            continue

        janela.freq.append((median_r_peaks_last, median_r_peaks, median_r_peaks - median_r_peaks_last))
        for i in freq_table.get_children():
            freq_table.delete(i)
        for f in janela.freq:
            freq_table.insert("", tk.END, values = f)
    
        janela.qrs.append((median_r_onsets, median_r_offsets, median_r_peaks - median_r_peaks_last, median_r_offsets - median_r_onsets))
        for i in qrs_table.get_children():
            qrs_table.delete(i)
        for q in janela.qrs:
            qrs_table.insert("", tk.END, values = q)

        janela.qt.append((median_r_onsets, median_t_offsets, median_r_peaks - median_r_peaks_last, median_t_offsets - median_r_onsets))
        for i in qt_table.get_children():
            qt_table.delete(i)
        for q in janela.qt:
            qt_table.insert("", tk.END, values = q)

    message_label.config(text = "Automatic Markings Completed Successfully.")
    message_label.update_idletasks()

def select_electrodes():

    # Exibe uma janela gráfica para o usuário selecionar os eletrodos ECG que deseja analisar.

    # Funcionalidade:
    #     - Cria uma janela (`Toplevel`) com uma lista interativa de eletrodos.
    #     - Permite múltiplas seleções na lista (`selectmode='extended'`).
    #     - Caso o usuário selecione "All", retorna todos os eletrodos padrão.
    #     - Caso contrário, retorna apenas os eletrodos escolhidos pelo usuário.

    # Eletrodos disponíveis:
    #     - 'All', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'

    # Retorno:
    #     selected_electrode (list of str): Lista com os nomes dos eletrodos selecionados.
    #                                       Se "All" for escolhido, retorna todos os eletrodos padrão.

    electrodes_window = tk.Toplevel(janela)
    electrodes_window.title("Select Electrodes")

    electrode_list = ttk.Treeview(electrodes_window, columns=('Electrode'), show='headings', selectmode='extended')
    electrode_list.heading('Electrode', text='Electrode')
    electrode_list.pack(fill=tk.BOTH, expand=True)

    electrodes = ['All','I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for electrode in electrodes:
        electrode_list.insert("", tk.END, values=(electrode,))

    selected_electrode = []

    def on_select():
        selected_items = electrode_list.selection()
        if selected_items:
            for selected_item in selected_items:
                item = electrode_list.item(selected_item)
                selected_electrode.append(item['values'][0]) 
        electrodes_window.destroy()

    select_button = tk.Button(electrodes_window, text="Select", command=on_select)
    select_button.pack(pady=10)

    electrodes_window.wait_window()

    for signal in selected_electrode:
        if (signal == "All"):
            return ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    return selected_electrode 

def key_press(event):

    # Trata eventos de pressionamento de tecla na interface gráfica para registrar ou cancelar marcações manuais 
    # de intervalos no sinal (como períodos, QRS, QT, extrassístoles e arritmias).

    # Parâmetros:
    #     event: Objeto de evento do `matplotlib` contendo informações da tecla pressionada.

    # Teclas e ações:
    #     - 'F': Salva o intervalo atual como um "período" (RR) e insere na tabela `freq_table`.
    #     - 'R': Salva o intervalo como um complexo QRS na tabela `qrs_table`.
    #     - 'T': Salva o intervalo como um QT na tabela `qt_table`.
    #     - 'E': Salva o intervalo como uma extrassístole na tabela `extrasystole_table`.
    #     - 'A': Salva o intervalo como uma arritmia na tabela `arrhythmia_table`.
    #     - 'C': Cancela a marcação atual (remove linhas verticais e reseta estado).
    #     - 'Esc': Cancela todas as marcações visuais e reseta o estado da janela.

    # Funcionalidade:
    #     - Apenas executa ações se houver duas marcações feitas no gráfico (`janela.click_state == 2`).
    #     - As marcações são feitas previamente via cliques, armazenadas em `janela.line_coords`.
    #     - Cada tecla insere os valores formatados nas listas apropriadas da estrutura `janela` e atualiza as tabelas Tkinter.
    #     - Atualiza o rótulo de mensagem (`message_label`) com feedback textual ao usuário.
    #     - Remove marcações gráficas usando `ax.lines` e `fig.canvas.draw()` quando necessário.

    if event.keysym.lower() == 'f':
        if janela.click_state == 2:
            initial_x = f"{janela.line_coords[-2]:.2f}"
            final_x = f"{janela.line_coords[-1]:.2f}"
            interval = f"{janela.interval:.2f}"
            janela.freq.append((initial_x, final_x, interval))
            for i in freq_table.get_children():
                freq_table.delete(i)
            for f in janela.freq:
                freq_table.insert("", tk.END, values = f)
            message_label.config(text = "Period added successfully.")
            janela.click_state = 0
            janela.line_coords =  []
            dx_var.set("dx: 0.00")
    elif event.keysym.lower() == 'c':
        for line in ax.lines:
            if line.get_label() in ['vertical_line_1', 'vertical_line_2']:
                line.remove()
        janela.click_state = 0
        janela.line_coords =  []
        dx_var.set("dx: 0.00")
        for line in ax.lines:
            if line.get_label() in ['vertical_line_1', 'vertical_line_2']:
                line.remove()
        fig.canvas.draw()
        message_label.config(text = "")
    elif event.keysym.lower() == 'r':
        if janela.click_state == 2:
            initial_x = f"{janela.line_coords[-2]:.2f}"
            final_x = f"{janela.line_coords[-1]:.2f}"
            interval = f"{janela.interval:.2f}"
            freq = select_frequency()            
            if freq:
                janela.qrs.append((initial_x, final_x, freq, interval))
                for i in qrs_table.get_children():
                    qrs_table.delete(i)
                for q in janela.qrs:
                    qrs_table.insert("", tk.END, values = q)
                message_label.config(text = "QRS added successfully.")
                janela.click_state = 0
                dx_var.set("dx: 0.00")
                janela.line_coords = []
    elif event.keysym.lower() == 't':
        if janela.click_state == 2:
            initial_x = f"{janela.line_coords[-2]:.2f}"
            final_x = f"{janela.line_coords[-1]:.2f}"
            interval = f"{janela.interval:.2f}"
            freq = select_frequency()            
            if freq:
                janela.qt.append((initial_x, final_x, freq, interval))
                for i in qt_table.get_children():
                    qt_table.delete(i)
                for q in janela.qt:
                    qt_table.insert("", tk.END, values = q)
                message_label.config(text = "QT added successfully.")
                janela.click_state = 0
                dx_var.set("dx: 0.00")
                janela.line_coords = []
    elif event.keysym.lower() == 'e':
        if janela.click_state == 2:
            initial_x = f"{janela.line_coords[-2]:.2f}"
            final_x = f"{janela.line_coords[-1]:.2f}"
            interval = f"{janela.interval:.2f}"
            freq = select_frequency()            
            if freq:
                janela.extrasystole.append((initial_x, final_x, freq, interval))
                for i in extrasystole_table.get_children():
                    extrasystole_table.delete(i)
                for q in janela.extrasystole:
                    extrasystole_table.insert("", tk.END, values = q)
                message_label.config(text = "Extrasystole added successfully.")
                janela.click_state = 0
                dx_var.set("dx: 0.00")
                janela.line_coords = []
    elif event.keysym.lower() == 'a':
        if janela.click_state == 2:
            initial_x = f"{janela.line_coords[-2]:.2f}"
            final_x = f"{janela.line_coords[-1]:.2f}"
            interval = f"{janela.interval:.2f}"
            freq = select_frequency()            
            if freq:
                janela.arrhythmia.append((initial_x, final_x, freq, interval))
                for i in arrhythmia_table.get_children():
                    arrhythmia_table.delete(i)
                for q in janela.arrhythmia:
                    arrhythmia_table.insert("", tk.END, values = q)
                message_label.config(text = "Arrhythmia added successfully.")
                janela.click_state = 0
                dx_var.set("dx: 0.00")
                janela.line_coords = []
    elif event.keysym.lower() == 'escape':
        for line in ax.lines:
            if line.get_label() in ['vertical_line_1', 'vertical_line_2', 'qrs_1', 'qrs_2', 'freq_1', 'freq_2', 'qt_1', 'qt_2', 'extrasystole_1', 'extrasystole_2', 'arrhythmia_1', 'arrhythmia_2']:
                line.remove()
        fig.canvas.draw()
        message_label.config(text = "")
        janela.click_state = 0
        dx_var.set("dx: 0.00")
        janela.line_coords = []

def select_frequency():

    # Abre uma janela modal para o usuário selecionar um período (frequência) a partir de uma lista 
    # pré-existente de períodos exibidos na tabela `freq_table`.

    # Funcionalidade:
    #     - Cria uma nova janela (`Toplevel`) contendo uma `Treeview` com colunas para:
    #       - Ponto inicial (Initial X)
    #       - Ponto final (Final X)
    #       - Período (Frequency)
    #     - Preenche a lista com os valores atuais da tabela global `freq_table`.
    #     - Permite seleção única de um período.
    #     - Ao clicar em "Select", a janela é fechada e o período selecionado é retornado.

    # Retorno:
    #     selected_freq (float ou None): Valor do período selecionado pelo usuário (coluna 'frequency').
    #                                    Retorna `None` se nenhuma seleção for feita.

    freq_window = tk.Toplevel(janela)
    freq_window.title("Select Period")
    
    freq_list = ttk.Treeview(freq_window, columns=('initial_x', 'final_x', 'frequency'), show='headings')
    freq_list.heading('initial_x', text = 'Initial X')
    freq_list.heading('final_x', text = 'Final X')
    freq_list.heading('frequency', text = 'Period')
    freq_list.pack(fill = tk.BOTH, expand = True)
    
    for child in freq_table.get_children():
        item = freq_table.item(child)
        freq_list.insert("", tk.END, values = item['values'])
    
    selected_freq = []
    
    def on_select():
        selected_item = freq_list.selection()
        if selected_item:
            item = freq_list.item(selected_item)
            selected_freq.append(item['values'][2])
            freq_window.destroy()
    
    select_button = tk.Button(freq_window, text = "Select", command=on_select)
    select_button.pack(pady=10)
    
    freq_window.wait_window()
    return selected_freq[0] if selected_freq else None

def freq_selected(event):

    # Evento acionado ao selecionar um intervalo na tabela de frequências (`freq_table`).

    # Funcionalidade:
    #     - Remove linhas verticais verdes existentes relacionadas à frequência no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais verdes no gráfico,
    #       representando o início e o fim do período selecionado.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (freq_table).

    for line in ax.lines:
        if line.get_label() in ['freq_1', 'freq_2']:
            line.remove()
    for selected_freq in freq_table.selection():
        item = freq_table.item(selected_freq)
        record = item['values']
        initial_x = float(record[0])
        final_x = float(record[1])
        ax.axvline(initial_x, color = 'green', linestyle = '-', label = 'freq_1', linewidth = 1)
        ax.axvline(final_x, color = 'green', linestyle = '-', label = 'freq_2', linewidth = 1)
    fig.canvas.draw()

def qrs_selected(event):

    # Evento acionado ao selecionar um intervalo na tabela de complexos QRS (`qrs_table`).

    # Funcionalidade:
    #     - Remove linhas verticais roxas existentes relacionadas ao QRS no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais roxas no gráfico,
    #       indicando início e fim do complexo QRS.
    #     - Atualiza o gráfico para exibir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (qrs_table).

    for line in ax.lines:
        if line.get_label() in ['qrs_1', 'qrs_2']:
            line.remove()
    for selected_qrs in qrs_table.selection():
        item = qrs_table.item(selected_qrs)
        record = item['values']
        initial_x = float(record[0])
        final_x = float(record[1])
        ax.axvline(initial_x, color = 'purple', linestyle = '-', label = 'qrs_1', linewidth = 1)
        ax.axvline(final_x, color = 'purple', linestyle = '-', label = 'qrs_2', linewidth = 1)
    fig.canvas.draw()

def qt_selected(event):

    # Evento acionado ao selecionar um intervalo na tabela QT (`qt_table`).

    # Funcionalidade:
    #     - Remove linhas verticais laranjas existentes relacionadas ao intervalo QT no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais laranjas no gráfico,
    #       representando início e fim do intervalo QT.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (qt_table).

    for line in ax.lines:
        if line.get_label() in ['qt_1', 'qt_2']:
            line.remove()
    for selected_qt in qt_table.selection():
        item = qt_table.item(selected_qt)
        record = item['values']
        initial_x = float(record[0])
        final_x = float(record[1])
        ax.axvline(initial_x, color = 'orange', linestyle = '-', label = 'qt_1', linewidth = 1)
        ax.axvline(final_x, color = 'orange', linestyle = '-', label = 'qt_2', linewidth = 1)
    fig.canvas.draw()

def extrasystole_selected(event):

    # Evento acionado ao selecionar um intervalo na tabela de extrassístoles (`extrasystole_table`).

    # Funcionalidade:
    #     - Remove linhas verticais amarelas existentes relacionadas às extrassístoles no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais amarelas no gráfico,
    #       indicando início e fim da extrassístole.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (extrasystole_table).

    for line in ax.lines:
        if line.get_label() in ['extrasystole_1', 'extrasystole_2']:
            line.remove()
    for selected_extrasystole in extrasystole_table.selection():
        item = extrasystole_table.item(selected_extrasystole)
        record = item['values']
        initial_x = float(record[0])
        final_x = float(record[1])
        ax.axvline(initial_x, color = 'yellow', linestyle = '-', label = 'extrasystole_1', linewidth = 1)
        ax.axvline(final_x, color = 'yellow', linestyle = '-', label = 'extrasystole_2', linewidth = 1)
    fig.canvas.draw()

def arrhythmia_selected(event):

    # Evento acionado ao selecionar um intervalo na tabela de arritmias (`arrhythmia_table`).

    # Funcionalidade:
    #     - Remove linhas verticais rosas existentes relacionadas às arritmias no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais rosas no gráfico,
    #       indicando início e fim da arritmia.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (arrhythmia_table).

    for line in ax.lines:
        if line.get_label() in ['arrhythmia_1', 'arrhythmia_2']:
            line.remove()
    for selected_arrhythmia in arrhythmia_table.selection():
        item = arrhythmia_table.item(selected_arrhythmia)
        record = item['values']
        initial_x = float(record[0])
        final_x = float(record[1])
        ax.axvline(initial_x, color = 'pink', linestyle = '-', label = 'arrhythmia_1', linewidth = 1)
        ax.axvline(final_x, color = 'pink', linestyle = '-', label = 'arrhythmia_2', linewidth = 1)
    fig.canvas.draw()

def save_data():

    # Salva os dados extraídos da análise do ECG em arquivos de texto e gráficos no diretório especificado.

    # Funcionalidade:
    #     - Cria o diretório de saída (`output_dir`) se não existir.
    #     - Salva:
    #         - Número de linhas do sinal original.
    #         - Sinais dos eletrodos selecionados (exceto linhas verticais de marcação).
    #         - Dados anotados: períodos, QRS, QT, extrassístoles e arritmias.
    #     - Para cada tipo de marcação (QRS, QT, velocidade estimada, APD), gera e salva gráficos (`.png`) e arquivos `.txt` com os dados.

    # Arquivos gerados:
    #     - `<output_file>`: arquivo geral com todos os sinais e anotações.
    #     - `qrs_file`: arquivo com dados de Período × Duração do QRS.
    #     - `qt_file`: arquivo com dados de Período × Duração do QT.
    #     - `extrasystole_file`: dados das extrassístoles (período, duração e tempos).
    #     - `arrhythmia_file`: dados das arritmias (período, duração e tempos).
    #     - `vel_file`: estimativas de velocidade normalizada (1/duração do QRS).
    #     - `apd_file`: diferença entre QT e QRS (duração estimada do APD).
    #     - `QRS.png`, `QT.png`, `Velocity.png`, `APD.png`: gráficos gerados com `matplotlib`.

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + output_file, 'w') as f:
        f.write(f"Num Lines:\n")
        f.write(f"{num_lines}\n")
        f.write("Curves:\n")
        for line in ax.lines:
            if line.get_label() not in ['vertical_line_1', 'vertical_line_2', 'qrs_1', 'qrs_2', 'freq_1', 'freq_2', 'qt_1', 'qt_2', 'extrasystole_1', 'extrasystole_2', 'arrhythmia_1', 'arrhythmia_2']:
                f.write(f"{line.get_label()}:\n")
                for y in electrodes[line.get_label()]['values']:
                    f.write(f"{y}\n")      

        f.write("Period Data:\n")
        for item in freq_table.get_children():
            values = freq_table.item(item)['values']
            f.write(f"{values[0]}, {values[1]}, {values[2]}\n")
        
        f.write("QRS Data:\n")
        for item in qrs_table.get_children():
            values = qrs_table.item(item)['values']
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}\n")
        
        f.write("QT Data:\n")
        for item in qt_table.get_children():
            values = qt_table.item(item)['values']
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}\n")
        
        f.write("Extrasystole Data:\n")
        for item in extrasystole_table.get_children():
            values = extrasystole_table.item(item)['values']
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}\n")
        
        f.write("Arrhythmia Data:\n")
        for item in arrhythmia_table.get_children():
            values = arrhythmia_table.item(item)['values']
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}\n")

    plt.figure()
        
    with open(output_dir + qrs_file, 'w') as f:
        f.write("Period, QRS\n")
        for item in qrs_table.get_children():
            values = qrs_table.item(item)['values']
            f.write(f"{values[2]}, {values[3]}\n")
            plt.plot(float(values[2]), float(values[3]), 'bo')

    plt.xlabel("Period")
    plt.ylabel("QRS")
    plt.savefig("QRS.png")

    with open(output_dir + extrasystole_file, 'w') as f:
        f.write("Period, Duration, Initial Time, Final Time\n")
        for item in extrasystole_table.get_children():
            values = extrasystole_table.item(item)['values']
            f.write(f"{values[2]}, {values[3]}, {values[0]}, {values[1]}\n")

    with open(output_dir + arrhythmia_file, 'w') as f:
        f.write("Period, Duration, Initial Time, Final Time\n")
        for item in arrhythmia_table.get_children():
            values = arrhythmia_table.item(item)['values']
            f.write(f"{values[2]}, {values[3]}, {values[0]}, {values[1]}\n")

    plt.figure()

    with open(output_dir + qt_file, 'w') as f:
        f.write("Period, QT\n")
        for item in qt_table.get_children():
            values = qt_table.item(item)['values']
            f.write(f"{values[2]}, {values[3]}\n")
            plt.plot(float(values[2]), float(values[3]), 'bo')

    plt.xlabel("Period")
    plt.ylabel("QT")
    plt.savefig("QT.png")

    estimated_normalized_velocity = []

    for qrs in janela.qrs:
        estimated_normalized_velocity.append(1 / float(qrs[3]))

    max_value = max(estimated_normalized_velocity)

    for indice in range(len(estimated_normalized_velocity)):
        estimated_normalized_velocity[indice] /= max_value

    plt.figure()
    
    with open(output_dir + vel_file, 'w') as f:
        f.write("Period, Estimated Normalized Velocity\n")
        for indice in range(len(estimated_normalized_velocity)):
            f.write(f"{janela.qrs[indice][2]}, {estimated_normalized_velocity[indice]}\n")
            plt.plot(janela.qrs[indice][2], estimated_normalized_velocity[indice], 'bo')
    
    plt.xlabel("Period")
    plt.ylabel("Estimated Normalized Velocity")
    plt.savefig("Velocity.png")

    plt.figure()

    with open(output_dir + apd_file, 'w') as f:
        f.write("Period, Estimated APD\n")
        if (len(janela.qt) == len(janela.qrs)):
            tam = len(janela.qt)
        for indice in range(tam):
            if (janela.qrs[indice][2] == janela.qt[indice][2]):
                period = janela.qt[indice][2]
                f.write(f"{float(period)}, {float(janela.qt[indice][3]) - float(janela.qrs[indice][3])}\n")
                plt.plot(float(period), float(janela.qt[indice][3]) - float(janela.qrs[indice][3]), 'bo')

    plt.xlabel("Period")
    plt.ylabel("Estimated APD")
    plt.savefig("APD.png")

    message_label.config(text = "Data saved successfully.")

def read_data(input_file):

    # Lê e interpreta os dados salvos em um arquivo de entrada formatado, contendo informações de sinais e anotações.

    # Estrutura esperada do arquivo:
    #     - "Num Lines:" seguido do número de linhas (amostras) por sinal.
    #     - "Curves:" seguido por blocos com nome do eletrodo e valores (um por linha).
    #     - "Period Data:" com linhas contendo (initial_x, final_x, interval).
    #     - "QRS Data:" com linhas contendo (initial_x, final_x, period, qrs_duration).
    #     - "QT Data:" com linhas contendo (initial_x, final_x, period, qt_duration).
    #     - "Extrasystole Data:" com linhas contendo (initial_x, final_x, period, duration).
    #     - "Arrhythmia Data:" com linhas contendo (initial_x, final_x, period, duration).

    # Parâmetros:
    #     input_file (str): Caminho para o arquivo de texto contendo os dados salvos.

    # Retorna:
    #     - infos (dict): Dicionário com os sinais dos eletrodos. Formato:
    #           {
    #             'I': {'values': [v1, v2, ...]},
    #             ...
    #           }
    #     - num_lines (int): Número de linhas (amostras) por eletrodo.
    #     - freq_data (list of tuples): Dados de períodos. Ex: [(ini, fim, intervalo), ...]
    #     - qrs_data (list of tuples): Dados de complexos QRS. Ex: [(ini, fim, periodo, duracao), ...]
    #     - qt_data (list of tuples): Dados de intervalos QT. Ex: [(ini, fim, periodo, duracao), ...]
    #     - extrasystole_data (list of tuples): Dados de extrassístoles. Ex: [(ini, fim, periodo, duracao), ...]
    #     - arrhythmia_data (list of tuples): Dados de arritmias. Ex: [(ini, fim, periodo, duracao), ...]
        
    with open(input_file, 'r') as f:
        data = f.read().splitlines()

    freq_data = []
    qrs_data = []
    qt_data = []
    extrasystole_data = []
    arrhythmia_data = []

    num_lines = 0
    infos = {}

    ind = -1

    section = None
    for line in data:
        if line.startswith("Num Lines:"):
            section = "Num Lines"
            continue
        if line.startswith("Curves:"):
            section = "Curves"
            continue
        elif line.startswith("Period Data:"):
            section = "Period Data"
            continue
        elif line.startswith("QRS Data:"):
            section = "QRS Data"
            continue
        elif line.startswith("QT Data:"):
            section = "QT Data"
            continue
        elif line.startswith("Extrasystole Data:"):
            section = "Extrasystole Data"
            continue
        elif line.startswith("Arrhythmia Data:"):
            section = "Arrhythmia Data"
            continue
        else:
            if section == "Num Lines":
                num_lines = int(line)
            elif section == "Curves":
                if line.split(":")[0] in head:
                    infos[line.split(":")[0]] = {'values': []}
                    ind += 1
                else:
                    y_data = line
                    infos[head[ind]]['values'].append(float(y_data))
            elif section == "Period Data":
                initial_x, final_x, interval = line.split(",")
                freq_data.append((initial_x, final_x, interval))
            elif section == "QRS Data":
                initial_x, final_x, interval, qrs = line.split(",")
                qrs_data.append((initial_x, final_x, interval, qrs))
            elif section == "QT Data":
                initial_x, final_x, interval, qt = line.split(",")
                qt_data.append((initial_x, final_x, interval, qt))
            elif section == "Extrasystole Data":
                initial_x, final_x, interval, duration = line.split(",")
                extrasystole_data.append((initial_x, final_x, interval, duration))
            elif section == "Arrhythmia Data":
                initial_x, final_x, interval, duration = line.split(",")
                arrhythmia_data.append((initial_x, final_x, interval, duration))

    return infos, num_lines, freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data

def update_tables(freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data):

    # Atualiza as tabelas gráficas da interface e as listas internas da janela com os dados fornecidos.

    # Parâmetros:
    #     freq_data (list of tuples): Lista de períodos (início, fim, duração).
    #     qrs_data (list of tuples): Lista de complexos QRS (início, fim, período, duração).
    #     qt_data (list of tuples): Lista de intervalos QT (início, fim, período, duração).
    #     extrasystole_data (list of tuples): Lista de extrassístoles (início, fim, período, duração).
    #     arrhythmia_data (list of tuples): Lista de arritmias (início, fim, período, duração).

    # Ações:
    #     - Insere os dados nas tabelas visuais correspondentes (`freq_table`, `qrs_table`, etc.).
    #     - Atualiza os atributos da `janela` com os novos dados.

    janela.freq = freq_data
    for f in janela.freq:
        freq_table.insert("", tk.END, values = f)
    janela.qrs = qrs_data
    for q in janela.qrs:
        qrs_table.insert("", tk.END, values = q)
    janela.qt = qt_data
    for q in janela.qt:
        qt_table.insert("", tk.END, values = q)
    janela.extrasystole = extrasystole_data
    for q in janela.extrasystole:
        extrasystole_table.insert("", tk.END, values = q)
    janela.arrhythmia = arrhythmia_data
    for q in janela.arrhythmia:
        arrhythmia_table.insert("", tk.END, values = q)

def delete_selected(event):

    # Remove o(s) item(ns) selecionado(s) da tabela correspondente e da estrutura de dados interna.

    # Parâmetros:
    #     event: Evento gerado ao pressionar a tecla Delete ou botão correspondente.
    #            Atributo `event.widget` é usado para identificar qual tabela foi ativada.

    # Tabelas suportadas:
    #     - freq_table: períodos
    #     - qrs_table: QRS
    #     - qt_table: QT
    #     - extrasystole_table: extrassístoles
    #     - arrhythmia_table: arritmias

    # Ações:
    #     - Identifica a tabela associada ao evento.
    #     - Remove os itens selecionados visualmente da tabela e logicamente da lista correspondente na `janela`.
    #     - Exibe mensagem de sucesso via `message_label`.

    table = -1
    selected_item = None
    if event.widget == freq_table:
        selected_item = freq_table.selection()
        table = 1
    elif event.widget == qrs_table:
        selected_item = qrs_table.selection()
        table = 2
    elif event.widget == qt_table:
        selected_item = qt_table.selection()
        table = 3
    elif event.widget == extrasystole_table:
        selected_item = extrasystole_table.selection()
        table = 4
    elif event.widget == arrhythmia_table:
        selected_item = arrhythmia_table.selection()
        table = 5

    if selected_item:
        for item in selected_item:
            if table == 1:
                item_index = freq_table.index(item)
                del janela.freq[item_index]
            if table == 2:
                item_index = qrs_table.index(item)
                del janela.qrs[item_index]
            if table == 3:
                item_index = qt_table.index(item)
                del janela.qt[item_index]
            if table == 4:
                item_index = extrasystole_table.index(item)
                del janela.extrasystole[item_index]
            if table == 5:
                item_index = arrhythmia_table.index(item)
                del janela.arrhythmia[item_index]
            event.widget.delete(item)
        message_label.config(text="Item deleted successfully.")

def plot_data():    

    # Cria uma janela gráfica (Tkinter) com dois gráficos:
    # - Velocidade Normalizada Estimada vs. Período.
    # - APD Estimada vs. Período (se disponível).

    # Ações:
    #     - Calcula a velocidade normalizada a partir da inversa da duração do QRS.
    #     - Normaliza os valores pela velocidade máxima.
    #     - Plota os pontos da curva "Velocidade x Período".
    #     - Se o número de entradas em `janela.qt` e `janela.qrs` for igual, calcula e plota o APD (QT - QRS).
    #     - Salva o gráfico de velocidade como "Velocity.png".
    #     - Insere os gráficos na janela com `FigureCanvasTkAgg`.

    plot_window = tk.Toplevel(janela)
    plot_window.title("Graphics")
    
    fig_qrs, ax_qrs = plt.subplots()
    ax_qrs.set_xlabel('Period')
    ax_qrs.set_ylabel('Estimated Normalized Velocity')
    ax_qrs.set_title("Estimated Normalized Velocity x Period")

    estimated_normalized_velocity = []

    for qrs in janela.qrs:
        estimated_normalized_velocity.append(1 / float(qrs[3]))

    max_value = max(estimated_normalized_velocity)

    for indice in range(len(estimated_normalized_velocity)):
        estimated_normalized_velocity[indice] /= max_value
    
    for indice in range(len(estimated_normalized_velocity)):
        ax_qrs.plot(float(janela.qrs[indice][2]), estimated_normalized_velocity[indice], 'ro')
    
    ax_qrs.legend()
    fig_qrs.savefig("Velocity.png")
    
    canvas_qrs = tkagg.FigureCanvasTkAgg(fig_qrs, master=plot_window)
    canvas_qrs.get_tk_widget().pack()

    if (len(janela.qt) == len(janela.qrs)):

        tam = len(janela.qt)

        fig_qt_corrected, ax_qt_corrected = plt.subplots()
        ax_qt_corrected.set_xlabel('Period')
        ax_qt_corrected.set_ylabel('Estimated APD')
        ax_qt_corrected.set_title("Estimated APD x Period")
        
        for indice in range(tam):
            if (janela.qrs[indice][2] == janela.qt[indice][2]):
                period = janela.qt[indice][2]
                ax_qt_corrected.plot(float(period), float(janela.qt[indice][3]) - float(janela.qrs[indice][3]), 'bo')
            else:
                print('Lack of equivalence between QRS and QT periods')
        
        ax_qt_corrected.legend()
        
        canvas_qt_corrected = tkagg.FigureCanvasTkAgg(fig_qt_corrected, master=plot_window)
        canvas_qt_corrected.get_tk_widget().pack()
    else:
        print('Lack of equivalence between QRS and QT periods')
                
def onmotion(event):

    # Atualiza dinamicamente o valor de dx (diferença entre o ponto atual e o ponto de clique anterior)
    # enquanto o cursor se move sobre o gráfico.

    # Parâmetros:
    #     event: Evento de movimento do mouse. Deve conter `xdata` e `inaxes`.

    # Ações:
    #     - Verifica se o mouse está sobre o eixo `ax`.
    #     - Se `janela.click_state == 1`, calcula e exibe a distância horizontal (dx) entre a posição atual do mouse
    #       e o último ponto de clique armazenado em `janela.line_coords`.

    if event.inaxes == ax:
        if janela.click_state == 1:
            x_atual = event.xdata
            diferenca = abs(x_atual - janela.line_coords[-1])
            dx_var.set(f"dx: {diferenca:.2f}")

def ecg_marker():

    global janela, fig, ax, xlim, freq_table, qrs_table, qt_table, extrasystole_table, arrhythmia_table, dx_var
    global message_label, scrollbar, num_lines, electrodes, clean_signal, output_dir, output_file, qrs_file, qt_file
    global apd_file, vel_file, arrhythmia_file, extrasystole_file, raw_data, input_file, textbox

    parser = argparse.ArgumentParser(description = 'ECG Marker')
    parser.add_argument('-i', action = 'store', dest = 'input', required = False, help = 'Input')
    parser.add_argument('-f', action = 'store', dest = 'input_file', required = False, default = 1, help = 'Input File (1) or Input Directory (0)')
    parser.add_argument('-d', action = 'store', dest = 'output_dir', required = False, default = './output/', help = 'Output Directory')
    parser.add_argument('-o', action = 'store', dest = 'output_file', required = False, default = 'ecg_data.txt', help = 'Output File')
    parser.add_argument('--qrs_file', action = 'store', dest = 'qrs_file', required = False, default = 'qrs_file.txt', help = 'Output file with QRS data')
    parser.add_argument('--qt_file', action = 'store', dest = 'qt_file', required = False, default = 'qt_file.txt', help = 'Output file with QT data')
    parser.add_argument('--vel_file', action = 'store', dest = 'vel_file', required = False, default = 'vel_file.txt', help = 'Output file with estimated normalized velocity data')
    parser.add_argument('--arrhythmia_file', action = 'store', dest = 'arrhythmia_file', required = False, default = 'arrhythmia_file.txt', help = 'Output file with arrhythmia marking')
    parser.add_argument('--extrasystole_file', action = 'store', dest = 'extrasystole_file', required = False, default = 'extrasystole_file.txt', help = 'Output file with extrasystole marking')
    parser.add_argument('--apd_file', action = 'store', dest = 'apd_file', required = False, default = 'apd_file.txt', help = 'Output file with estimated APD data')
    parser.add_argument('-r', action = 'store', dest = 'raw_data', required = False, default = 1, help = 'Raw Data (1) or not (0)')
    parser.add_argument('-c', action = 'store', dest = 'clean_signal', required = False, default = 0, help = 'Clean signal (1) or not (0)')

    arguments = parser.parse_args()

    input = arguments.input
    output_file = arguments.output_file
    output_dir = arguments.output_dir
    qrs_file = arguments.qrs_file
    qt_file = arguments.qt_file
    apd_file = arguments.apd_file
    vel_file = arguments.vel_file
    arrhythmia_file = arguments.arrhythmia_file
    extrasystole_file = arguments.extrasystole_file
    raw_data = int(arguments.raw_data)
    input_file = int(arguments.input_file)
    clean_signal = int(arguments.clean_signal)

    electrodes = {}
    num_lines = 0

    if raw_data:
        if input_file:
            electrodes, num_lines = read_file(input)
        else:
            electrodes, num_lines = read_dir(input)
    else:
        electrodes, num_lines, freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data = read_data(input)
            
    janela = tk.Tk()
    janela.title("ECG Marker")

    janela.columnconfigure(0, weight = 10)
    janela.columnconfigure(1, weight = 1)
    janela.rowconfigure(0, weight = 1)

    screen_width = janela.winfo_screenwidth() - 12
    screen_height = janela.winfo_screenheight() - 92

    janela.geometry(f"{screen_width}x{screen_height}")

    x = np.arange(0, num_lines)

    offset = 0
    cont   = 0

    xlim = 1000

    fig, ax = plt.subplots()
    ax.set_xlim(0, xlim)

    for electrode in electrodes:
        if (cont != 0):
            offset += 1000
            # offset += max(electrodes[electrode]['values']) - min(electrodes[electrode]['values'])
        ax.plot(x, np.array(electrodes[electrode]['values']) + offset, label = electrode)
        # offset += max(electrodes[electrode]['values'])
        cont += 1
    ax.legend(loc = 'upper left')
    ax.yaxis.set_visible(False)
    ax.set_xlabel('t')

    scrollbar_ax = plt.axes([0.13, 0.02, 0.8, 0.03], facecolor = 'lightgoldenrodyellow')
    scrollbar = widgets.Slider(scrollbar_ax, 'Eixo X', 0, num_lines, valinit = 0)

    scrollbar.on_changed(update)

    frame_left = tk.Frame(janela)
    frame_left.grid(row = 0, column = 0, sticky = "nsew", rowspan = 6)

    canvas_widget = tkagg.FigureCanvasTkAgg(fig, master = frame_left)
    canvas_widget.get_tk_widget().pack(fill = tk.BOTH, expand = True)

    toolbar = tkagg.NavigationToolbar2Tk(canvas_widget, frame_left)
    toolbar.update()

    message_label = tk.Label(janela, text = "", font = ('Arial', 12), background='white')
    message_label.grid(row = 0, column = 0, sticky = 'n', pady = 40, padx = 20)

    frame_right = tk.Frame(janela)
    frame_right.grid(row = 0, column = 1, sticky = "nsew")
    frame_right.columnconfigure(0, weight = 1)
    frame_right.columnconfigure(1, weight = 1)
    frame_right.rowconfigure(0, weight = 1)
    frame_right.rowconfigure(1, weight = 1)
    frame_right.rowconfigure(2, weight = 3)
    frame_right.rowconfigure(3, weight = 1)
    frame_right.rowconfigure(4, weight = 3)
    frame_right.rowconfigure(5, weight = 1)
    frame_right.rowconfigure(6, weight = 3)
    frame_right.rowconfigure(7, weight = 1)
    frame_right.rowconfigure(8, weight = 3)
    frame_right.rowconfigure(9, weight = 1)
    frame_right.rowconfigure(10, weight = 3)
    frame_right.rowconfigure(11, weight = 1)

    freq_button = tk.Button(frame_right, text='Automatic Marking', command = automatic_period_marking)
    freq_button.grid(row = 0, column = 2, columnspan = 4, padx = 20, pady = 10, ipadx = 20)

    freq_name = tk.Label(frame_right, text = "Period", font = ('Arial', 16))
    freq_name.grid(column = 0, row = 1, columnspan = 4, padx = 2, pady = 2)

    freq_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency'), show = 'headings', height = 5)
    freq_table.grid(row = 2, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    freq_table.heading('initial_x', text = 'Initial X')
    freq_table.heading('final_x', text = 'Final X')
    freq_table.heading('frequency', text = 'Period')

    freq_table.column('initial_x', width = 160, anchor = 'center')
    freq_table.column('final_x', width = 160, anchor = 'center')
    freq_table.column('frequency', width = 160, anchor = 'center')

    scrollbar_vertical_freq = tk.Scrollbar(frame_right, orient='vertical', command=freq_table.yview)
    scrollbar_vertical_freq.grid(row=2, column=4, sticky='ns')
    freq_table.configure(yscrollcommand=scrollbar_vertical_freq.set)

    freq_table.bind('<<TreeviewSelect>>', freq_selected)

    qrs_name = tk.Label(frame_right, text = "QRS", font = ('Arial', 16))
    qrs_name.grid(column = 0, row = 3, columnspan = 4, padx = 2, pady = 2)

    qrs_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'qrs'), show = 'headings', height = 5)
    qrs_table.grid(row = 4, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    qrs_table.heading('initial_x', text = 'Initial X')
    qrs_table.heading('final_x', text = 'Final X')
    qrs_table.heading('frequency', text = 'Period')
    qrs_table.heading('qrs', text = 'QRS')

    qrs_table.column('initial_x', width = 120, anchor = 'center')
    qrs_table.column('final_x', width = 120, anchor = 'center')
    qrs_table.column('frequency', width = 120, anchor = 'center')
    qrs_table.column('qrs', width = 120, anchor = 'center')

    scrollbar_vertical_qrs = tk.Scrollbar(frame_right, orient='vertical', command=qrs_table.yview)
    scrollbar_vertical_qrs.grid(row=4, column=4, sticky='ns')
    qrs_table.configure(yscrollcommand=scrollbar_vertical_qrs.set)

    qrs_table.bind('<<TreeviewSelect>>', qrs_selected)

    qt_name = tk.Label(frame_right, text = "QT", font = ('Arial', 16))
    qt_name.grid(column = 0, row = 5, columnspan = 4, padx = 2, pady = 2)

    qt_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'qt'), show = 'headings', height = 5)
    qt_table.grid(row = 6, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    qt_table.heading('initial_x', text = 'Initial X')
    qt_table.heading('final_x', text = 'Final X')
    qt_table.heading('frequency', text = 'Period')
    qt_table.heading('qt', text = 'QT')

    qt_table.column('initial_x', width = 120, anchor = 'center')
    qt_table.column('final_x', width = 120, anchor = 'center')
    qt_table.column('frequency', width = 120, anchor = 'center')
    qt_table.column('qt', width = 120, anchor = 'center')

    scrollbar_vertical_qt = tk.Scrollbar(frame_right, orient='vertical', command=qt_table.yview)
    scrollbar_vertical_qt.grid(row=6, column=4, sticky='ns')
    qt_table.configure(yscrollcommand=scrollbar_vertical_qt.set)

    qt_table.bind('<<TreeviewSelect>>', qt_selected)

    extrasystole_name = tk.Label(frame_right, text = "Extrasystole", font = ('Arial', 16))
    extrasystole_name.grid(column = 0, row = 7, columnspan = 4, padx = 2, pady = 2)

    extrasystole_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'duration'), show = 'headings', height = 5)
    extrasystole_table.grid(row = 8, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    extrasystole_table.heading('initial_x', text = 'Initial X')
    extrasystole_table.heading('final_x', text = 'Final X')
    extrasystole_table.heading('frequency', text = 'Period')
    extrasystole_table.heading('duration', text = 'Duration')

    extrasystole_table.column('initial_x', width = 120, anchor = 'center')
    extrasystole_table.column('final_x', width = 120, anchor = 'center')
    extrasystole_table.column('frequency', width = 120, anchor = 'center')
    extrasystole_table.column('duration', width = 120, anchor = 'center')

    scrollbar_vertical_extrasystole = tk.Scrollbar(frame_right, orient='vertical', command=extrasystole_table.yview)
    scrollbar_vertical_extrasystole.grid(row=8, column=4, sticky='ns')
    extrasystole_table.configure(yscrollcommand=scrollbar_vertical_extrasystole.set)

    extrasystole_table.bind('<<TreeviewSelect>>', extrasystole_selected)

    arrhythmia_name = tk.Label(frame_right, text = "Arrhythmia", font = ('Arial', 16))
    arrhythmia_name.grid(column = 0, row = 9, columnspan = 4, padx = 2, pady = 2)

    arrhythmia_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'duration'), show = 'headings', height = 5)
    arrhythmia_table.grid(row = 10, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    arrhythmia_table.heading('initial_x', text = 'Initial X')
    arrhythmia_table.heading('final_x', text = 'Final X')
    arrhythmia_table.heading('frequency', text = 'Period')
    arrhythmia_table.heading('duration', text = 'Duration')

    arrhythmia_table.column('initial_x', width = 120, anchor = 'center')
    arrhythmia_table.column('final_x', width = 120, anchor = 'center')
    arrhythmia_table.column('frequency', width = 120, anchor = 'center')
    arrhythmia_table.column('duration', width = 120, anchor = 'center')

    scrollbar_vertical_arrhythmia = tk.Scrollbar(frame_right, orient='vertical', command=arrhythmia_table.yview)
    scrollbar_vertical_arrhythmia.grid(row=10, column=4, sticky='ns')
    arrhythmia_table.configure(yscrollcommand=scrollbar_vertical_arrhythmia.set)

    arrhythmia_table.bind('<<TreeviewSelect>>', arrhythmia_selected)

    freq_table.bind('<Delete>', delete_selected)
    qrs_table.bind('<Delete>', delete_selected)
    qt_table.bind('<Delete>', delete_selected)
    extrasystole_table.bind('<Delete>', delete_selected)
    arrhythmia_table.bind('<Delete>', delete_selected)

    username_label = tk.Label(frame_right, text = "X Limit:")
    username_label.grid(column = 0, row = 12, padx = 10, pady = 1, ipadx = 1, ipady = 1, sticky = 'e')

    textbox = tk.Entry(frame_right)
    textbox.insert(tk.END, xlim)
    textbox.grid(row = 12, column = 1, padx = 0, pady = 0, sticky='w')
    textbox.bind('<Return>', on_enter)

    plot_button = tk.Button(frame_right, text='Plot Data', command = plot_data)
    plot_button.grid(row = 12, column = 2, pady = 10, ipadx = 10)

    save = tk.Button(frame_right, text = 'Save', command = save_data)
    save.grid(row = 12, column = 3, padx = 20, pady = 10, ipadx = 20)

    janela.click_state = 0
    janela.line_coords = []

    janela.freq = []
    janela.qrs = []
    janela.qt = []
    janela.extrasystole = []
    janela.arrhythmia = []

    dx_var = tk.StringVar()

    dx_var.set("dx: 0.00")

    dx = tk.Label(janela, textvariable=dx_var)
    dx.grid(column = 0, row = 11, padx = 10, pady = 1, ipadx = 1, ipady = 1, sticky = 'e')

    fig.canvas.mpl_connect('motion_notify_event', onmotion)

    if (not raw_data):
        update_tables(freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data)

    janela.bind('<Key>', key_press)

    fig.canvas.mpl_connect('button_press_event', onclick)

    janela.mainloop()

# if __name__ == "__main__":
#     ecg_marker()