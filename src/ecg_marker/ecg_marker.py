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
import re
import configparser
import ast
import sys

# ecg_nn lives at the repo root, two levels up from this file (src/ecg_marker/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ecg_nn import Recording as _NNRecording

DEFAULT_HEAD_FILE    = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'HISp', 'HISd', 'VD p', 'VD 78', 'VD 56', 'VD 34', 'VD d']
DEFAULT_HEAD         = ['VD d', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DEFAULT_HEAD_MONO    = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DEFAULT_INPUT        = "./input/"
DEFAULT_INPUT_FILE   = 0
DEFAULT_OUTPUT_DIR   = "./output/"
DEFAULT_RAW_DATA     = 1
DEFAULT_CLEAN_SIGNAL = 0
DEFAULT_ECG_MONO     = 0
DEFAULT_OFFSET       = 1000
DEFAULT_UNCERTAINTY  = 15.0

# ecg_nn: how automatic_period_marking() handles beats whose window overlaps
# a neighbor's (see ecg_nn.recording.NOISY_BEAT_MODES for the full contract).
# Changeable at runtime via the "ecg_nn ⚙" toolbar button.
NOISY_BEAT_MODE_INFO = {
    'recovery': ("Recovery (same as training)",
                 "Rescues overlapping beats with a shifted, truncated window."),
    'exclude':  ("Exclude",
                 "Skips inference for any beat whose window overlaps a neighbor."),
    'force':    ("Force",
                 "Runs inference on every beat, using its plain R-peak-centered window regardless of overlap."),
}
DEFAULT_NOISY_BEAT_MODE = 'recovery'
noisy_beat_mode = DEFAULT_NOISY_BEAT_MODE

# Which production QRS ensemble bundle to use (see ecg_nn.recording.ENSEMBLE_BUNDLES
# for the full contract). Only takes effect if the bundle files are present under
# models/production/ -- otherwise ecg_nn falls back to the single mid-training checkpoint.
ENSEMBLE_BUNDLE_INFO = {
    '4fold':    ("4-fold",
                 "32-member ensemble from 4 leave-one-out folds x 8 seeds."),
    'light':    ("Light",
                 "4-member ensemble, one seed per leave-one-out fold -- faster, less averaging."),
    'complete': ("Complete",
                 "64-member ensemble trained on all patients with no holdout."),
}
DEFAULT_ENSEMBLE_BUNDLE = '4fold'
ensemble_bundle = DEFAULT_ENSEMBLE_BUNDLE

# Configuração de cada tabela de marcação: em qual posição da tupla estão os campos
# 'initial' (tempo inicial), 'final' (tempo final), 'duration' (duração/intervalo, recalculada
# automaticamente ao arrastar uma marcação), 'uncertainty' (incerteza, em ms) e os campos
# 'anchor_initial' / 'anchor_final' (os valores ORIGINAIS de início/fim no momento em que a
# marcação foi criada — não são exibidos na tabela, mas são usados para fixar os limites
# (mínimo/máximo) da faixa de incerteza no gráfico, que não se movem mesmo depois de a marcação
# ser reajustada por arraste). Além disso, indica qual atributo de `janela` guarda a lista de
# dados e qual variável global guarda o widget Treeview.
#
# Para as tabelas QRS, QT, Extrasystole e Arrhythmia, a tupla tem ainda uma 8ª posição
# (índice 7) chamada 'freq_ref': o índice da marcação de Período (em janela.freq) da qual o
# valor de 'frequency' (índice 2) foi copiado. Esse vínculo permite propagar automaticamente
# qualquer reajuste de período (feito por arraste na tabela de Período) para todas as tabelas
# que dependem dele, mesmo quando duas marcações de período têm o mesmo valor numérico.
# freq_ref é None quando o período foi digitado manualmente (sem vínculo com nenhuma marcação).
TABLE_FIELD_CONFIG = {
    'freq':         {'initial': 0, 'final': 1, 'duration': 2, 'uncertainty': 3, 'anchor_initial': 4, 'anchor_final': 5, 'list_attr': 'freq',         'table_name': 'freq_table'},
    # 'uncertainty' (index 4) is the ONSET/start uncertainty; 'uncertainty_end' (index 8,
    # appended after freq_ref so old files / other marking types are unaffected) is the
    # OFFSET/end uncertainty -- for ecg_nn v6 QRS marks these are the model's own
    # per-beat tau_on/tau_off, genuinely different. Optional: draw_marking_with_band
    # falls back to symmetric (uncertainty for both ends) when it's absent, which is
    # what manual 'R' entries and older saved files produce.
    'qrs':          {'initial': 0, 'final': 1, 'duration': 3, 'uncertainty': 4, 'anchor_initial': 5, 'anchor_final': 6, 'uncertainty_end': 8, 'list_attr': 'qrs', 'table_name': 'qrs_table'},
    'qt':           {'initial': 0, 'final': 1, 'duration': 3, 'uncertainty': 4, 'anchor_initial': 5, 'anchor_final': 6, 'list_attr': 'qt',           'table_name': 'qt_table'},
    'extrasystole': {'initial': 0, 'final': 1, 'duration': 3, 'uncertainty': 4, 'anchor_initial': 5, 'anchor_final': 6, 'list_attr': 'extrasystole', 'table_name': 'extrasystole_table'},
    'arrhythmia':   {'initial': 0, 'final': 1, 'duration': 3, 'uncertainty': 4, 'anchor_initial': 5, 'anchor_final': 6, 'list_attr': 'arrhythmia',   'table_name': 'arrhythmia_table'},
}

# Tabelas cujo campo 'frequency' (índice 2) referencia uma marcação de Período através do
# campo freq_ref (índice 7). São essas as tabelas atualizadas quando um Período é arrastado
# ou removido.
PERIOD_DEPENDENT_TABLES = ('qrs', 'qt', 'extrasystole', 'arrhythmia')
PERIOD_FIELD_INDEX = 2
FREQ_REF_INDEX = 7

def get_table_widget (marking_type):
    # Retorna o widget Treeview correspondente ao tipo de marcação informado.
    return globals()[TABLE_FIELD_CONFIG[marking_type]['table_name']]

def make_sort_handler (marking_type, column):
    
    # Cria (e retorna) uma função de callback para ser usada como `command` do cabeçalho de uma
    # coluna de uma Treeview, permitindo ordenar a tabela pelo valor daquela coluna (ex.: incerteza).
    # A cada clique no cabeçalho, a ordem é invertida (ascendente / descendente).
    #
    # ATENÇÃO: ordenar a tabela de Período (freq) invalida os índices freq_ref guardados nas
    # tabelas dependentes (QRS, QT, Extrasystole, Arrhythmia), já que esses índices assumem a
    # ordem original de janela.freq. Por isso, ordenar por incerteza na tabela de Período
    # também reindexa os vínculos nas tabelas dependentes.

    def handler ():
        cfg = TABLE_FIELD_CONFIG[marking_type]
        data_list = getattr(janela, cfg['list_attr'])
        table_widget = get_table_widget(marking_type)
        state_key = f"{marking_type}_{column}"
        reverse = janela.sort_state.get(state_key, False)
        field_index = cfg[column]

        if marking_type == 'freq':
            # Guarda a ordem antiga (índice -> id do objeto) para poder remapear os freq_ref
            # das tabelas dependentes depois de ordenar.
            old_order = list(data_list)
            indexed = list(enumerate(data_list))
            indexed.sort(key = lambda pair: float(pair[1][field_index]), reverse = reverse)
            new_order = [row for _, row in indexed]
            old_index_to_new_index = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(indexed)}

            data_list[:] = new_order

            for i in table_widget.get_children():
                table_widget.delete(i)
            for row in data_list:
                table_widget.insert("", tk.END, values = row)

            remap_freq_refs(old_index_to_new_index)
        else:
            data_list.sort(key = lambda row: float(row[field_index]), reverse = reverse)

            for i in table_widget.get_children():
                table_widget.delete(i)
            for row in data_list:
                table_widget.insert("", tk.END, values = row)

        janela.sort_state[state_key] = not reverse

    return handler

def remap_freq_refs (old_index_to_new_index):

    # Atualiza o campo freq_ref (índice 7) em todas as tabelas dependentes de período após a
    # tabela de Período ter sido reordenada, usando o mapeamento {índice_antigo: índice_novo}.

    for marking_type in PERIOD_DEPENDENT_TABLES:
        cfg = TABLE_FIELD_CONFIG[marking_type]
        data_list = getattr(janela, cfg['list_attr'])
        table_widget = get_table_widget(marking_type)
        children = table_widget.get_children()

        for i, row in enumerate(data_list):
            freq_ref = row[FREQ_REF_INDEX] if len(row) > FREQ_REF_INDEX else None
            if freq_ref is None or freq_ref not in old_index_to_new_index:
                continue
            new_row = list(row)
            new_row[FREQ_REF_INDEX] = old_index_to_new_index[freq_ref]
            data_list[i] = tuple(new_row)
            if i < len(children):
                table_widget.item(children[i], values = data_list[i])

def clear_markers (line_labels, band_labels):

    # Remove do gráfico as linhas verticais e as faixas sombreadas (bandas de incerteza)
    # associadas aos rótulos (labels) informados.

    for line in list(ax.lines):
        if line.get_label() in line_labels:
            line.remove()
    for patch in list(ax.patches):
        if patch.get_label() in band_labels:
            patch.remove()

def clear_draggable (marking_type):

    # Remove do registro de linhas arrastáveis (`janela.draggable_lines`) todas as entradas
    # associadas ao tipo de marcação informado (chamado antes de redesenhar a seleção de uma tabela).

    for line in list(janela.draggable_lines.keys()):
        if janela.draggable_lines[line]['type'] == marking_type:
            del janela.draggable_lines[line]

def draw_marking_with_band (values, index, marking_type, color, label1, label2, band_label1, band_label2):

    # Desenha, para uma marcação selecionada em uma tabela, duas linhas verticais (início e fim)
    # e, ao redor de cada uma, uma faixa sombreada representando a incerteza da marcação
    # (valor ± metade da incerteza). As linhas são registradas como arrastáveis: o usuário pode
    # clicar sobre uma delas e arrastá-la, ficando o movimento restrito aos limites da faixa.

    # Parâmetros:
    #     values: valores da linha da tabela (initial_x, final_x, ..., uncertainty).
    #     index (int): posição da marcação na lista de dados correspondente (janela.<tipo>).
    #     marking_type (str): chave em TABLE_FIELD_CONFIG.
    #     color (str): cor usada para as linhas e a faixa.
    #     label1, label2 (str): labels das linhas verticais (início / fim).
    #     band_label1, band_label2 (str): labels das faixas sombreadas (início / fim).

    cfg = TABLE_FIELD_CONFIG[marking_type]
    initial_x = float(values[cfg['initial']])
    final_x = float(values[cfg['final']])
    uncertainty = float(values[cfg['uncertainty']])
    half = uncertainty / 2.0

    # Onset/offset uncertainty band width: symmetric (half from the single
    # 'uncertainty' field) unless this marking type has a distinct end-side
    # value (currently only QRS, index 8 -- appended after freq_ref so it's
    # absent from manual entries and pre-existing saved files, which fall
    # back to the symmetric case here).
    end_uncertainty_idx = cfg.get('uncertainty_end')
    if end_uncertainty_idx is not None and len(values) > end_uncertainty_idx:
        half_end = float(values[end_uncertainty_idx]) / 2.0
    else:
        half_end = half

    # Os limites da faixa são calculados a partir dos valores ORIGINAIS (âncora), fixados no
    # momento em que a marcação foi criada — não a partir da posição atual da linha — para que
    # a faixa não se desloque quando a marcação for reajustada por arraste.
    anchor_initial = float(values[cfg['anchor_initial']])
    anchor_final = float(values[cfg['anchor_final']])
    initial_low, initial_high = anchor_initial - half, anchor_initial + half
    final_low, final_high = anchor_final - half_end, anchor_final + half_end

    line1 = ax.axvline(initial_x, color = color, linestyle = '-', label = label1, linewidth = 1)
    line2 = ax.axvline(final_x, color = color, linestyle = '-', label = label2, linewidth = 1)

    band1 = ax.axvspan(initial_low, initial_high, color = color, alpha = 0.15, label = band_label1)
    band2 = ax.axvspan(final_low, final_high, color = color, alpha = 0.15, label = band_label2)

    # 'half'/'color'/'band_label' + 'band_patch' let on_release_drag recenter the band
    # around wherever the user drops the line (see that function) -- the band's WIDTH
    # never changes, only its center, and each release moves it a bit further.
    janela.draggable_lines[line1] = {'type': marking_type, 'index': index, 'field': 'initial',
                                      'low': initial_low, 'high': initial_high, 'half': half,
                                      'color': color, 'band_label': band_label1, 'band_patch': band1}
    janela.draggable_lines[line2] = {'type': marking_type, 'index': index, 'field': 'final',
                                      'low': final_low, 'high': final_high, 'half': half_end,
                                      'color': color, 'band_label': band_label2, 'band_patch': band2}

def try_start_drag (event):

    # Verifica se o clique do mouse ocorreu próximo a uma linha de marcação arrastável.
    # Se sim, inicia o modo de arraste (armazenando a linha e seus limites) e retorna True.
    # Caso contrário, retorna False, permitindo que o clique seja tratado normalmente por `onclick`.

    if not janela.draggable_lines or event.xdata is None:
        return False

    xlim_range = ax.get_xlim()
    tolerance = (xlim_range[1] - xlim_range[0]) * 0.006

    for line, info in janela.draggable_lines.items():
        if line not in ax.lines:
            continue
        x0 = line.get_xdata()[0]
        if abs(event.xdata - x0) <= tolerance:
            janela.dragging_line = line
            janela.dragging_info = info
            return True

    return False

def update_drag (event):

    # Atualiza a posição da linha de marcação sendo arrastada, restringindo o movimento aos
    # limites da faixa de incerteza (`info['low']` a `info['high']`) associada a essa linha.

    if janela.dragging_line is None or event.xdata is None:
        return

    info = janela.dragging_info
    new_x = min(max(event.xdata, info['low']), info['high'])
    janela.dragging_line.set_xdata([new_x, new_x])
    fig.canvas.draw_idle()

def on_release_drag (event):

    # Finaliza a operação de arraste: confirma a nova posição da marcação, atualizando a lista
    # de dados interna e a linha correspondente na tabela (Treeview).
    #
    # Também RECENTRA a faixa de incerteza na posição onde o mouse foi solto: a largura
    # (info['half']) não muda, só o centro. Isso vale tanto para os novos limites 'low'/'high'
    # (usados por update_drag para restringir o próximo arraste) quanto para a faixa sombreada
    # desenhada no gráfico -- o patch antigo é removido e um novo é desenhado na posição nova.
    # Resultado: soltar o mouse uma vez trava o arraste seguinte nessa faixa recentrada, mas o
    # usuário pode soltar de novo (e de novo) para continuar avançando além da faixa original.

    if janela.dragging_line is None:
        return

    info = janela.dragging_info
    new_x = janela.dragging_line.get_xdata()[0]
    commit_drag_value(info['type'], info['index'], info['field'], new_x)

    half = info['half']
    new_low, new_high = new_x - half, new_x + half
    info['low'] = new_low
    info['high'] = new_high

    old_patch = info.get('band_patch')
    if old_patch is not None:
        old_patch.remove()
    info['band_patch'] = ax.axvspan(new_low, new_high, color = info['color'], alpha = 0.15,
                                     label = info['band_label'])

    janela.dragging_line = None
    janela.dragging_info = None
    fig.canvas.draw_idle()

def commit_drag_value (marking_type, index, field, new_x):

    # Grava o novo valor (tempo inicial ou final) de uma marcação após o usuário arrastar a linha
    # dentro da faixa de incerteza, recalculando o campo de duração/intervalo e atualizando a
    # linha correspondente na tabela (Treeview).
    #
    # Também recentra a âncora (anchor_initial/anchor_final) do lado arrastado na nova posição --
    # ao contrário do comportamento anterior (âncora fixa no valor de criação), isso permite que
    # cada soltura do mouse desloque a faixa de incerteza, possibilitando arrastar progressivamente
    # além da faixa original em soltadas sucessivas (ver on_release_drag, que também reposiciona a
    # faixa sombreada e os limites 'low'/'high' correspondentes).
    #
    # Quando a marcação arrastada é do tipo 'freq' (Período), o novo valor de duração
    # (o período recalculado) é propagado para todas as tabelas dependentes de período
    # (QRS, QT, Extrasystole, Arrhythmia) que estejam vinculadas a essa marcação via freq_ref.

    cfg = TABLE_FIELD_CONFIG[marking_type]
    data_list = getattr(janela, cfg['list_attr'])
    if index >= len(data_list):
        return

    row = list(data_list[index])
    field_index = cfg['initial'] if field == 'initial' else cfg['final']
    row[field_index] = f"{new_x:.2f}"

    anchor_field_index = cfg['anchor_initial'] if field == 'initial' else cfg['anchor_final']
    row[anchor_field_index] = f"{new_x:.2f}"

    initial_x = float(row[cfg['initial']])
    final_x = float(row[cfg['final']])
    row[cfg['duration']] = f"{abs(final_x - initial_x):.2f}"

    data_list[index] = tuple(row)

    table_widget = get_table_widget(marking_type)
    children = table_widget.get_children()
    if index < len(children):
        table_widget.item(children[index], values = data_list[index])

    if marking_type == 'freq':
        update_linked_periods(index, row[cfg['duration']])

def update_linked_periods (freq_index, new_period):

    # Atualiza o campo 'frequency' (índice 2) em todas as tabelas dependentes de período
    # (QRS, QT, Extrasystole, Arrhythmia) cujo freq_ref (índice 7) aponte para o índice de
    # freq que acabou de ser arrastado. O vínculo é por índice — não por valor — para que
    # marcações de Período com o mesmo valor numérico não sejam confundidas entre si.

    for marking_type in PERIOD_DEPENDENT_TABLES:
        cfg = TABLE_FIELD_CONFIG[marking_type]
        data_list = getattr(janela, cfg['list_attr'])
        table_widget = get_table_widget(marking_type)
        children = table_widget.get_children()

        for i, row in enumerate(data_list):
            freq_ref = row[FREQ_REF_INDEX] if len(row) > FREQ_REF_INDEX else None
            if freq_ref != freq_index:
                continue
            new_row = list(row)
            new_row[PERIOD_FIELD_INDEX] = new_period
            data_list[i] = tuple(new_row)
            if i < len(children):
                table_widget.item(children[i], values = data_list[i])

def cascade_freq_deletion (deleted_index):

    # Ao remover uma marcação da tabela de Período, atualiza o campo freq_ref (índice 7) em
    # todas as tabelas dependentes de período: marcações que apontavam exatamente para o
    # índice removido perdem o vínculo (freq_ref = None, o valor de 'frequency' permanece como
    # estava, apenas deixa de ser atualizado automaticamente); marcações que apontavam para
    # índices posteriores têm o índice decrementado em 1, para continuar apontando para a
    # marcação de Período correta após a remoção.

    for marking_type in PERIOD_DEPENDENT_TABLES:
        cfg = TABLE_FIELD_CONFIG[marking_type]
        data_list = getattr(janela, cfg['list_attr'])
        table_widget = get_table_widget(marking_type)
        children = table_widget.get_children()

        for i, row in enumerate(data_list):
            freq_ref = row[FREQ_REF_INDEX] if len(row) > FREQ_REF_INDEX else None
            if freq_ref is None:
                continue
            if freq_ref == deleted_index:
                new_ref = None
            elif freq_ref > deleted_index:
                new_ref = freq_ref - 1
            else:
                continue
            new_row = list(row)
            new_row[FREQ_REF_INDEX] = new_ref
            data_list[i] = tuple(new_row)
            if i < len(children):
                table_widget.item(children[i], values = data_list[i])

def read_config (config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    head_file         = DEFAULT_HEAD_FILE.copy()
    head              = DEFAULT_HEAD.copy()
    head_mono         = DEFAULT_HEAD_MONO.copy()
    input             = DEFAULT_INPUT
    input_file        = DEFAULT_INPUT_FILE
    output_dir        = DEFAULT_OUTPUT_DIR
    raw_data          = DEFAULT_RAW_DATA
    clean_signal      = DEFAULT_CLEAN_SIGNAL
    ecg_mono          = DEFAULT_ECG_MONO
    offset            = DEFAULT_OFFSET
    uncertainty_value = DEFAULT_UNCERTAINTY

    if config.has_section('electrodes'):
        if config.has_option('electrodes', 'head_file'):
            try:
                head_file = ast.literal_eval(config.get('electrodes', 'head_file'))
            except (ValueError, SyntaxError):
                print("Aviso: 'head_file' inválido. Usando valor padrão.")

        if config.has_option('electrodes', 'head'):
            try:
                head = ast.literal_eval(config.get('electrodes', 'head'))
            except (ValueError, SyntaxError):
                print("Aviso: 'head' inválido. Usando valor padrão.")

    if config.has_section('data'):
        if config.has_option('data', 'input'):
            try:
                input = config.get('data', 'input')
            except (ValueError, SyntaxError):
                print("Aviso: 'input' inválido. Usando valor padrão.")

        if config.has_option('data', 'input_file'):
            try:
                input_file = config.getint('data', 'input_file')
            except ValueError:
                print("Aviso: 'input_file' inválido. Usando valor padrão.")

        if config.has_option('data', 'output_dir'):
            try:
                output_dir = config.get('data', 'output_dir')
            except (ValueError, SyntaxError):
                print("Aviso: 'output_dir' inválido. Usando valor padrão.")

        if config.has_option('data', 'raw_data'):
            try:
                raw_data = config.getint('data', 'raw_data')
            except ValueError:
                print("Aviso: 'raw_data' inválido. Usando valor padrão.")

    if config.has_section('marking'):
        if config.has_option('marking', 'clean_signal'):
            try:
                clean_signal = config.getint('marking', 'clean_signal')
            except ValueError:
                print("Aviso: 'clean_signal' inválido. Usando valor padrão.")

        if config.has_option('marking', 'ecg_mono'):
            try:
                ecg_mono = config.getint('marking', 'ecg_mono')
            except ValueError:
                print("Aviso: 'ecg_mono' inválido. Usando valor padrão.")

        if config.has_option('marking', 'offset'):
            try:
                offset = config.getint('marking', 'offset')
            except ValueError:
                print("Aviso: 'offset' inválido. Usando valor padrão.")

        if config.has_option('marking', 'uncertainty'):
            try:
                uncertainty_value = config.getint('marking', 'uncertainty')
            except ValueError:
                print("Aviso: 'uncertainty' inválido. Usando valor padrão.")

    if ecg_mono:
        head_mono = head.copy()

    return head_file, head, head_mono, input, input_file, output_dir, raw_data, clean_signal, ecg_mono, offset, uncertainty_value

def read_file (filename):

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

def read_dir (input_dir):

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

def read_dir_2 (input_dir):

    # Lê todos os arquivos .txt de um diretório contendo sinais de eletrodos (formato com colunas separadas por espaço)
    # e agrega os dados em um único dicionário.

    # Parâmetros:
    #     input_dir (str): Caminho para o diretório contendo os arquivos .txt.
    #     head_mono (list): Lista com os nomes dos eletrodos, na ordem das colunas (após o tempo).

    # Retorno:
    #     infos (dict): Dicionário contendo os valores dos eletrodos especificados em head_mono.
    #                   Estrutura: { 'V1': {'values': [...]}, 'V2': {'values': [...]}, ... }
    #     num_lines (int): Número total de linhas de dados somando todos os arquivos.

    infos = {electrode: {'values': []} for electrode in head_mono}
    num_lines = 0

    arquivos = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    arquivos.sort(key=lambda x: int(re.match(r'(\d+)', x).group()))  # pega o número no início

    for filename in arquivos:

        print(filename)

        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            lines = f.readlines()
            lines = lines[::5]
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()  # separa por espaços (qualquer quantidade)

                if len(parts) < len(head_mono) + 1:
                    continue  # linha inválida

                # Ignora a primeira coluna (tempo)
                values = parts[1:]

                for i, electrode in enumerate(head_mono):
                    infos[electrode]['values'].append(float(values[i]))

            num_lines += len(lines)

    return infos, num_lines

def update (val):

    # Atualiza a visualização do gráfico com base na posição da barra de rolagem.

    # Parâmetros:
    #     val (float): Novo valor da barra de rolagem (posição inicial da janela de visualização).

    start_index = int(scrollbar.val)

    if (start_index + xlim < num_lines):
        ax.set_xlim(start_index, start_index + xlim)
    else:
        ax.set_xlim(start_index - xlim, start_index)

    fig.canvas.draw_idle()

def center_view_on (x_center):

    # Centra a visualização (mesma largura atual, `xlim`) em torno de x_center, e sincroniza a
    # barra de rolagem para refletir a nova posição -- usado quando exatamente UMA marcação está
    # selecionada, para que o zoom atual do usuário não seja alterado.

    start_index = x_center - xlim / 2.0
    start_index = max(0, min(start_index, max(0, num_lines - xlim)))
    scrollbar.set_val(start_index)

    fig.canvas.draw_idle()

def fit_view_to_range (range_min, range_max):

    # Ajusta o ZOOM (largura da janela de visualização, `xlim`) para que a faixa
    # [range_min, range_max] fique inteiramente visível, com uma margem de folga em cada lado,
    # e sincroniza a barra de rolagem -- usado quando MAIS DE UMA marcação está selecionada
    # simultaneamente, para que todas fiquem visíveis ao mesmo tempo (ao contrário da seleção
    # única, aqui o zoom atual é intencionalmente alterado).

    global xlim
    span = max(range_max - range_min, 1.0)
    margin = max(span * 0.1, 20.0)  # at least 20ms of breathing room on each side
    xlim = min(span + 2 * margin, num_lines)

    start_index = range_min - margin
    start_index = max(0, min(start_index, max(0, num_lines - xlim)))
    scrollbar.set_val(start_index)

    fig.canvas.draw_idle()

def center_or_fit_view (values_list, cfg):

    # Ponto único de decisão entre as duas funções acima: uma marcação selecionada -> centraliza
    # sem alterar o zoom (center_view_on); mais de uma -> ajusta o zoom para mostrar todas
    # (fit_view_to_range), usando o mínimo de 'initial' e o máximo de 'final' entre as selecionadas.

    # Parâmetros:
    #     values_list: lista de tuplas `values` (uma por linha selecionada na tabela).
    #     cfg: TABLE_FIELD_CONFIG[marking_type] correspondente.

    if not values_list:
        return

    if len(values_list) == 1:
        v = values_list[0]
        center_view_on((float(v[cfg['initial']]) + float(v[cfg['final']])) / 2.0)
    else:
        all_initial = [float(v[cfg['initial']]) for v in values_list]
        all_final   = [float(v[cfg['final']]) for v in values_list]
        fit_view_to_range(min(all_initial), max(all_final))

def on_enter (event):

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

def onclick (event):

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

        # Se o clique for próximo a uma linha de marcação já plotada (dentro da faixa de
        # incerteza), inicia uma operação de arraste em vez de criar uma nova marcação.
        if try_start_drag(event):
            return

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

def open_ecg_nn_settings ():

    # Janela de configuração do ecg_nn: escolhe (1) o modo de tratamento de batimentos cujo
    # janela sobrepõe a de um vizinho (ver ecg_nn.recording.NOISY_BEAT_MODES e
    # NOISY_BEAT_MODE_INFO) e (2) qual bundle do ensemble de produção usar (ver
    # ecg_nn.recording.ENSEMBLE_BUNDLES e ENSEMBLE_BUNDLE_INFO). Os valores escolhidos ficam
    # em `noisy_beat_mode` / `ensemble_bundle` (globais) e são lidos por
    # automatic_period_marking() a cada execução.

    global noisy_beat_mode, ensemble_bundle

    win = tk.Toplevel(janela)
    win.title("ecg_nn Settings")

    tk.Label(win, text = "Noisy beat handling", font = ('Arial', 12, 'bold')).pack(anchor = 'w', padx = 12, pady = (12, 4))

    mode_var = tk.StringVar(value = noisy_beat_mode)
    for value in ('recovery', 'exclude', 'force'):
        label, desc = NOISY_BEAT_MODE_INFO[value]
        tk.Radiobutton(win, text = label, variable = mode_var, value = value, font = ('Arial', 10)).pack(anchor = 'w', padx = 12, pady = (8, 0))
        tk.Label(win, text = desc, font = ('Arial', 8), fg = 'gray30', justify = 'left').pack(anchor = 'w', padx = 34)

    tk.Label(win, text = "Model", font = ('Arial', 12, 'bold')).pack(anchor = 'w', padx = 12, pady = (16, 4))

    bundle_var = tk.StringVar(value = ensemble_bundle)
    for value in ('4fold', 'light', 'complete'):
        label, desc = ENSEMBLE_BUNDLE_INFO[value]
        tk.Radiobutton(win, text = label, variable = bundle_var, value = value, font = ('Arial', 10)).pack(anchor = 'w', padx = 12, pady = (8, 0))
        tk.Label(win, text = desc, font = ('Arial', 8), fg = 'gray30', justify = 'left').pack(anchor = 'w', padx = 34)

    def apply ():
        global noisy_beat_mode, ensemble_bundle
        noisy_beat_mode = mode_var.get()
        ensemble_bundle = bundle_var.get()
        message_label.config(text = f"Noisy-beat mode: {noisy_beat_mode}, model: {ensemble_bundle}")
        win.destroy()

    tk.Button(win, text = "Apply", command = apply).pack(pady = 14)

def automatic_period_marking ():

    # Marcação automática via a rede neural ecg_nn (MaskHeadV6 quando seu
    # checkpoint está presente, senão o FiLM MaskHead) -- substitui a
    # detecção antiga baseada em neurokit2 + mediana entre eletrodos.

    # Funcionalidade:
    #     - Roda sobre todos os eletrodos carregados de uma vez só (o encoder
    #       HuBERT do ecg_nn precisa do contexto das 12 derivações juntas, então
    #       não há mais seleção de eletrodo por marcação).
    #     - Para cada batimento detectado e não-ruidoso, adiciona uma linha de
    #       QRS (janela.qrs) a partir do onset/duração previstos -- um por
    #       R-peak, incluindo o primeiro. Período (RR) exige dois R-peaks, então
    #       o primeiro batimento não tem bcl (predecessor); nesse caso a linha
    #       de Período correspondente é omitida e a QRS fica com freq_ref=None
    #       (não vinculada), igual ao caso de período digitado manualmente.
    #     - Formato de tupla igual ao que os atalhos manuais 'F'/'R' produzem --
    #       assim o arraste manual continua funcionando normalmente.
    #     - QT não é preenchido: a inferência atual do ecg_nn não prevê
    #       intervalo QT (mesma limitação em ambos os modelos, FiLM e v6).

    # Block re-entry: a second click mid-run (e.g. because nothing seemed to
    # be happening yet) would kick off a second, overlapping inference pass.
    auto_mark_button['state'] = tk.DISABLED

    message_label.config(text = "Loading model")
    message_label.update_idletasks()

    leads = {name: np.array(data['values']) for name, data in electrodes.items()}

    progress_bar['value'] = 0
    progress_bar['mode'] = 'indeterminate'
    progress_bar.start(10)
    progress_bar.grid()
    progress_bar.lift()

    def _on_progress(stage, i, n):
        if stage == 'loading_model':
            message_label.config(text = "Loading model")
        elif stage == 'loading_encoder':
            message_label.config(text = "Loading HuBERT-ECG encoder")
        elif stage == 'inference':
            if str(progress_bar['mode']) != 'determinate':
                progress_bar.stop()
                progress_bar['mode'] = 'determinate'
            progress_bar['maximum'] = max(n, 1)
            progress_bar['value'] = i
            message_label.config(text = f"Running neural QRS detection: {i}/{n} beats")
        message_label.update_idletasks()
        progress_bar.update_idletasks()

    try:
        rec = _NNRecording.from_signal(leads, predict=True, progress_callback=_on_progress,
                                        noisy_beat_mode=noisy_beat_mode, ensemble_bundle=ensemble_bundle)
    except Exception as e:
        import traceback
        traceback.print_exc()
        message_label.config(text = f"Inference failed: {e}")
        return
    finally:
        progress_bar.stop()
        progress_bar.grid_remove()
        auto_mark_button['state'] = tk.NORMAL

    # 'force' mode means every beat, including ones flagged noisy (window
    # overlaps a neighbor), should show up in the marking tables -- 'recovery'
    # and 'exclude' both still want noisy beats kept out (see
    # open_ecg_nn_settings / NOISY_BEAT_MODE_INFO).
    skip_noisy = noisy_beat_mode != 'force'

    beats = rec.beats
    n_marked = 0
    for i, beat in enumerate(beats):
        if (skip_noisy and beat.noisy) or beat.qrs_start is None or beat.qrs_duration is None:
            continue

        # Period (RR) has no model output backing it -- it's pure spike-to-spike
        # arithmetic -- so its uncertainty stays the fixed config value.
        period_uncertainty = f"{uncertainty_value:.2f}"

        # QRS uncertainty: ecg_nn already derives real per-beat onset/offset
        # values and stores them on the beat -- use them instead of the
        # config constant. v6 is purely parametric: these ARE tau_on/tau_off,
        # its own learned uncertainty, read straight from the model (no mask
        # involved). FiLM (v4) has no such parameter, so this falls back to a
        # mask threshold-crossing width for that model. Falls back further to
        # the config value only if neither is available (e.g. FiLM's mask
        # never crossed threshold). Kept genuinely separate (not averaged/
        # maxed into one number) -- see TABLE_FIELD_CONFIG['qrs'].
        if beat.qrs_start_uncert is not None:
            qrs_uncertainty_on = f"{beat.qrs_start_uncert:.2f}"
        else:
            qrs_uncertainty_on = f"{uncertainty_value:.2f}"
        if beat.qrs_end_uncert is not None:
            qrs_uncertainty_off = f"{beat.qrs_end_uncert:.2f}"
        else:
            qrs_uncertainty_off = f"{uncertainty_value:.2f}"

        if beat.bcl is not None:
            # Normal case: this beat has a predecessor, so RR/period is defined.
            interval  = f"{beat.bcl:.2f}"
            initial_x = f"{(beat.spike_idx - beat.bcl):.2f}"
            final_x   = f"{beat.spike_idx:.2f}"

            # Índice que a marcação de Período recém-criada terá em janela.freq —
            # usado para vincular (via freq_ref) a marcação de QRS a ela.
            freq_idx = len(janela.freq)
            janela.freq.append((initial_x, final_x, interval, period_uncertainty, initial_x, final_x))
        else:
            # First beat: no predecessor, so no RR interval to anchor a Period
            # row to. Still emit its QRS mark (v6 gave it a real prediction) --
            # borrow the following beat's RR as a display estimate for the
            # 'frequency' column, same as a manually-typed, unlinked period.
            freq_idx = None
            interval = f"{beats[i + 1].bcl:.2f}" if (i + 1 < len(beats) and beats[i + 1].bcl is not None) else "0.00"

        qrs_start = f"{beat.qrs_start:.2f}"
        qrs_end   = f"{(beat.qrs_start + beat.qrs_duration):.2f}"
        qrs_dur   = f"{beat.qrs_duration:.2f}"
        janela.qrs.append((qrs_start, qrs_end, interval, qrs_dur, qrs_uncertainty_on,
                            qrs_start, qrs_end, freq_idx, qrs_uncertainty_off))

        n_marked += 1

    for i in freq_table.get_children():
        freq_table.delete(i)
    for f in janela.freq:
        freq_table.insert("", tk.END, values = f)

    for i in qrs_table.get_children():
        qrs_table.delete(i)
    for q in janela.qrs:
        qrs_table.insert("", tk.END, values = q)

    message_label.config(text = f"Automatic Markings Completed ({n_marked} beats).")
    message_label.update_idletasks()

def key_press (event):

    # Trata eventos de pressionamento de tecla na interface gráfica para registrar ou cancelar marcações manuais
    # de intervalos no sinal (como períodos, QRS, QT, extrassístoles e arritmias).

    # Parâmetros:
    #     event: Objeto de evento do `matplotlib` contendo informações da tecla pressionada.

    # Teclas e ações:
    #     - 'F': Salva o intervalo atual como um "período" (RR) e insere na tabela `freq_table`.
    #     - 'R': Salva o intervalo como um complexo QRS na tabela `qrs_table`, vinculado (via freq_ref)
    #            à marcação de Período escolhida em `select_frequency`.
    #     - 'T': Salva o intervalo como um QT na tabela `qt_table`, com o mesmo vínculo.
    #     - 'E': Salva o intervalo como uma extrassístole na tabela `extrasystole_table`, com o mesmo vínculo.
    #     - 'A': Salva o intervalo como uma arritmia na tabela `arrhythmia_table`, com o mesmo vínculo.
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
            uncertainty = f"{uncertainty_value:.2f}"
            janela.freq.append((initial_x, final_x, interval, uncertainty, initial_x, final_x))
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
            freq, freq_idx = select_frequency()
            if freq:
                uncertainty = f"{uncertainty_value:.2f}"
                # No model here to give a distinct offset uncertainty, so
                # duplicate the single value -- symmetric band either way
                # (see TABLE_FIELD_CONFIG['qrs']['uncertainty_end']).
                janela.qrs.append((initial_x, final_x, freq, interval, uncertainty, initial_x, final_x, freq_idx, uncertainty))
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
            freq, freq_idx = select_frequency()
            if freq:
                uncertainty = f"{uncertainty_value:.2f}"
                janela.qt.append((initial_x, final_x, freq, interval, uncertainty, initial_x, final_x, freq_idx))
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
            freq, freq_idx = select_frequency()
            if freq:
                uncertainty = f"{uncertainty_value:.2f}"
                janela.extrasystole.append((initial_x, final_x, freq, interval, uncertainty, initial_x, final_x, freq_idx))
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
            freq, freq_idx = select_frequency()
            if freq:
                uncertainty = f"{uncertainty_value:.2f}"
                janela.arrhythmia.append((initial_x, final_x, freq, interval, uncertainty, initial_x, final_x, freq_idx))
                for i in arrhythmia_table.get_children():
                    arrhythmia_table.delete(i)
                for q in janela.arrhythmia:
                    arrhythmia_table.insert("", tk.END, values = q)
                message_label.config(text = "Arrhythmia added successfully.")
                janela.click_state = 0
                dx_var.set("dx: 0.00")
                janela.line_coords = []
    elif event.keysym.lower() == 'escape':
        for line in list(ax.lines):
            if line.get_label() in ['vertical_line_1', 'vertical_line_2', 'qrs_1', 'qrs_2', 'freq_1', 'freq_2', 'qt_1', 'qt_2', 'extrasystole_1', 'extrasystole_2', 'arrhythmia_1', 'arrhythmia_2']:
                line.remove()
        for patch in list(ax.patches):
            if patch.get_label() in ['freq_band_1', 'freq_band_2', 'qrs_band_1', 'qrs_band_2', 'qt_band_1', 'qt_band_2', 'extrasystole_band_1', 'extrasystole_band_2', 'arrhythmia_band_1', 'arrhythmia_band_2']:
                patch.remove()
        janela.draggable_lines = {}
        janela.dragging_line = None
        janela.dragging_info = None
        freq_table.selection_remove(freq_table.selection())
        qrs_table.selection_remove(qrs_table.selection())
        qt_table.selection_remove(qt_table.selection())
        extrasystole_table.selection_remove(extrasystole_table.selection())
        arrhythmia_table.selection_remove(arrhythmia_table.selection())
        fig.canvas.draw()
        message_label.config(text = "")
        janela.click_state = 0
        dx_var.set("dx: 0.00")
        janela.line_coords = []

def select_frequency ():
    """
    Abre uma janela modal para o usuário selecionar um período (frequência) ou digitar manualmente.

    Retorno:
        (selected_freq, freq_idx): tupla com o valor do período (float) selecionado ou digitado,
        e o índice da marcação em janela.freq à qual ele está vinculado (None se digitado
        manualmente ou se nada foi selecionado). Se nada foi escolhido, retorna (None, None).
    """
    freq_window = tk.Toplevel(janela)
    freq_window.title("Select Period / Enter Manually")

    # Label + Entry para entrada manual
    tk.Label(freq_window, text="Digite manualmente um período (ms), ou selecione na lista:").pack(pady=5)
    manual_entry = tk.Entry(freq_window)
    manual_entry.pack(pady=5, fill=tk.X, padx=10)

    # Treeview com períodos existentes
    freq_list = ttk.Treeview(freq_window, columns=('initial_x', 'final_x', 'frequency'), show='headings')
    freq_list.heading('initial_x', text='Initial X')
    freq_list.heading('final_x', text='Final X')
    freq_list.heading('frequency', text='Period')
    freq_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # A ordem de freq_table.get_children() é sempre igual à ordem de janela.freq (ambas são
    # reconstruídas juntas em toda inserção/ordenação), então o índice `idx` aqui corresponde
    # diretamente ao índice em janela.freq. Ele é guardado oculto na 4ª coluna da lista, para
    # permitir vincular (freq_ref) a marcação escolhida.
    for idx, child in enumerate(freq_table.get_children()):
        item = freq_table.item(child)
        row_values = list(item['values'][:3]) + [idx]
        freq_list.insert("", tk.END, values = row_values)

    selected = []  # [(freq_value, freq_idx)]

    def on_select():
        # Verifica se algo foi digitado manualmente
        manual_value = manual_entry.get().strip()
        if manual_value != "":
            try:
                selected.append((float(manual_value), None))
                freq_window.destroy()
                return
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Digite um número válido.")
                return

        # Caso contrário, pega a seleção da lista
        sel = freq_list.selection()
        if sel:
            item = freq_list.item(sel)
            values = item['values']
            selected.append((float(values[2]), int(values[3])))
            freq_window.destroy()
        else:
            tk.messagebox.showwarning("No Selection", "Selecione um período ou digite manualmente.")

    select_button = tk.Button(freq_window, text="Select", command=on_select)
    select_button.pack(pady=10)

    freq_window.wait_window()
    return selected[0] if selected else (None, None)

def freq_selected (event):

    # Evento acionado ao selecionar um intervalo na tabela de frequências (`freq_table`).

    # Funcionalidade:
    #     - Remove linhas verticais verdes existentes relacionadas à frequência no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais verdes no gráfico,
    #       representando o início e o fim do período selecionado, além de uma faixa sombreada
    #       (banda de incerteza) ao redor de cada marcação, dentro da qual a marcação pode ser
    #       reajustada com o mouse (clicar e arrastar).
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (freq_table).

    clear_markers(['freq_1', 'freq_2'], ['freq_band_1', 'freq_band_2'])
    clear_draggable('freq')
    cfg = TABLE_FIELD_CONFIG['freq']
    values_list = []
    for selected_freq in freq_table.selection():
        item = freq_table.item(selected_freq)
        index = freq_table.index(selected_freq)
        draw_marking_with_band(item['values'], index, 'freq', 'green', 'freq_1', 'freq_2', 'freq_band_1', 'freq_band_2')
        values_list.append(item['values'])
    center_or_fit_view(values_list, cfg)
    fig.canvas.draw()

def qrs_selected (event):

    # Evento acionado ao selecionar um intervalo na tabela de complexos QRS (`qrs_table`).

    # Funcionalidade:
    #     - Remove linhas verticais roxas existentes relacionadas ao QRS no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais roxas no gráfico,
    #       indicando início e fim do complexo QRS, além da faixa sombreada de incerteza editável.
    #     - Atualiza o gráfico para exibir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (qrs_table).

    clear_markers(['qrs_1', 'qrs_2'], ['qrs_band_1', 'qrs_band_2'])
    clear_draggable('qrs')
    cfg = TABLE_FIELD_CONFIG['qrs']
    values_list = []
    for selected_qrs in qrs_table.selection():
        item = qrs_table.item(selected_qrs)
        index = qrs_table.index(selected_qrs)
        draw_marking_with_band(item['values'], index, 'qrs', 'purple', 'qrs_1', 'qrs_2', 'qrs_band_1', 'qrs_band_2')
        values_list.append(item['values'])
    center_or_fit_view(values_list, cfg)
    fig.canvas.draw()

def qt_selected (event):

    # Evento acionado ao selecionar um intervalo na tabela QT (`qt_table`).

    # Funcionalidade:
    #     - Remove linhas verticais laranjas existentes relacionadas ao intervalo QT no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais laranjas no gráfico,
    #       representando início e fim do intervalo QT, além da faixa sombreada de incerteza editável.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (qt_table).

    clear_markers(['qt_1', 'qt_2'], ['qt_band_1', 'qt_band_2'])
    clear_draggable('qt')
    cfg = TABLE_FIELD_CONFIG['qt']
    values_list = []
    for selected_qt in qt_table.selection():
        item = qt_table.item(selected_qt)
        index = qt_table.index(selected_qt)
        draw_marking_with_band(item['values'], index, 'qt', 'orange', 'qt_1', 'qt_2', 'qt_band_1', 'qt_band_2')
        values_list.append(item['values'])
    center_or_fit_view(values_list, cfg)
    fig.canvas.draw()

def extrasystole_selected (event):

    # Evento acionado ao selecionar um intervalo na tabela de extrassístoles (`extrasystole_table`).

    # Funcionalidade:
    #     - Remove linhas verticais amarelas existentes relacionadas às extrassístoles no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais amarelas no gráfico,
    #       indicando início e fim da extrassístole, além da faixa sombreada de incerteza editável.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (extrasystole_table).

    clear_markers(['extrasystole_1', 'extrasystole_2'], ['extrasystole_band_1', 'extrasystole_band_2'])
    clear_draggable('extrasystole')
    cfg = TABLE_FIELD_CONFIG['extrasystole']
    values_list = []
    for selected_extrasystole in extrasystole_table.selection():
        item = extrasystole_table.item(selected_extrasystole)
        index = extrasystole_table.index(selected_extrasystole)
        draw_marking_with_band(item['values'], index, 'extrasystole', 'yellow', 'extrasystole_1', 'extrasystole_2', 'extrasystole_band_1', 'extrasystole_band_2')
        values_list.append(item['values'])
    center_or_fit_view(values_list, cfg)
    fig.canvas.draw()

def arrhythmia_selected (event):

    # Evento acionado ao selecionar um intervalo na tabela de arritmias (`arrhythmia_table`).

    # Funcionalidade:
    #     - Remove linhas verticais rosas existentes relacionadas às arritmias no gráfico.
    #     - Para cada intervalo selecionado na tabela, adiciona duas linhas verticais rosas no gráfico,
    #       indicando início e fim da arritmia, além da faixa sombreada de incerteza editável.
    #     - Atualiza o gráfico para refletir essas linhas.

    # Parâmetros:
    #     event: Evento de seleção disparado pelo widget Treeview (arrhythmia_table).

    clear_markers(['arrhythmia_1', 'arrhythmia_2'], ['arrhythmia_band_1', 'arrhythmia_band_2'])
    clear_draggable('arrhythmia')
    cfg = TABLE_FIELD_CONFIG['arrhythmia']
    values_list = []
    for selected_arrhythmia in arrhythmia_table.selection():
        item = arrhythmia_table.item(selected_arrhythmia)
        index = arrhythmia_table.index(selected_arrhythmia)
        draw_marking_with_band(item['values'], index, 'arrhythmia', 'pink', 'arrhythmia_1', 'arrhythmia_2', 'arrhythmia_band_1', 'arrhythmia_band_2')
        values_list.append(item['values'])
    center_or_fit_view(values_list, cfg)
    fig.canvas.draw()

def _format_freq_ref (freq_ref):

    # Formata o campo freq_ref para gravação em disco: string vazia quando None (sem vínculo).

    return "" if freq_ref is None else str(freq_ref)

def save_data ():

    # Salva os dados extraídos da análise do ECG em arquivos de texto e gráficos no diretório especificado.

    # Funcionalidade:
    #     - Cria o diretório de saída (`output_dir`) se não existir.
    #     - Salva:
    #         - Número de linhas do sinal original.
    #         - Sinais dos eletrodos selecionados (exceto linhas verticais de marcação).
    #         - Dados anotados: períodos, QRS, QT, extrassístoles e arritmias (incluindo a incerteza de cada
    #           marcação e, para QRS/QT/Extrasystole/Arrhythmia, o índice de vínculo freq_ref com a marcação
    #           de Período correspondente, para que a propagação automática funcione ao reabrir o arquivo).
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
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}\n")

        f.write("QRS Data:\n")
        for item in qrs_table.get_children():
            values = qrs_table.item(item)['values']
            freq_ref = _format_freq_ref(values[FREQ_REF_INDEX] if len(values) > FREQ_REF_INDEX else None)
            # 9th field: offset-side uncertainty (see TABLE_FIELD_CONFIG['qrs']['uncertainty_end']);
            # defaults to the onset value (values[4]) for rows without it (manual entries always
            # have it now too, but stay defensive for anything hand-edited).
            uncertainty_end = values[8] if len(values) > 8 else values[4]
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}, {values[6]}, {freq_ref}, {uncertainty_end}\n")

        f.write("QT Data:\n")
        for item in qt_table.get_children():
            values = qt_table.item(item)['values']
            freq_ref = _format_freq_ref(values[FREQ_REF_INDEX] if len(values) > FREQ_REF_INDEX else None)
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}, {values[6]}, {freq_ref}\n")

        f.write("Extrasystole Data:\n")
        for item in extrasystole_table.get_children():
            values = extrasystole_table.item(item)['values']
            freq_ref = _format_freq_ref(values[FREQ_REF_INDEX] if len(values) > FREQ_REF_INDEX else None)
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}, {values[6]}, {freq_ref}\n")

        f.write("Arrhythmia Data:\n")
        for item in arrhythmia_table.get_children():
            values = arrhythmia_table.item(item)['values']
            freq_ref = _format_freq_ref(values[FREQ_REF_INDEX] if len(values) > FREQ_REF_INDEX else None)
            f.write(f"{values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]}, {values[6]}, {freq_ref}\n")

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

    # Skip non-positive durations (e.g. v6 predicting t_off < t_on, clamped to
    # 0 -- a real but invalid beat) instead of crashing on 1/0. Track period
    # alongside velocity since skipped entries shift indices out of sync with
    # janela.qrs.
    estimated_normalized_velocity = []
    velocity_periods = []

    for qrs in janela.qrs:
        duration = float(qrs[3])
        if duration <= 0:
            continue
        estimated_normalized_velocity.append(1 / duration)
        velocity_periods.append(qrs[2])

    if estimated_normalized_velocity:
        max_value = max(estimated_normalized_velocity)

        for indice in range(len(estimated_normalized_velocity)):
            estimated_normalized_velocity[indice] /= max_value

    plt.figure()

    with open(output_dir + vel_file, 'w') as f:
        f.write("Period, Estimated Normalized Velocity\n")
        for indice in range(len(estimated_normalized_velocity)):
            f.write(f"{velocity_periods[indice]}, {estimated_normalized_velocity[indice]}\n")
            plt.plot(velocity_periods[indice], estimated_normalized_velocity[indice], 'bo')

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

def read_data (input_file):

    # Lê e interpreta os dados salvos em um arquivo de entrada formatado, contendo informações de sinais e anotações.

    # Estrutura esperada do arquivo:
    #     - "Num Lines:" seguido do número de linhas (amostras) por sinal.
    #     - "Curves:" seguido por blocos com nome do eletrodo e valores (um por linha).
    #     - "Period Data:" com linhas contendo (initial_x, final_x, interval[, uncertainty[, anchor_initial, anchor_final]]).
    #     - "QRS Data:" com linhas contendo (initial_x, final_x, period, qrs_duration[, uncertainty[, anchor_initial, anchor_final[, freq_ref]]]).
    #     - "QT Data:" com o mesmo formato de "QRS Data".
    #     - "Extrasystole Data:" com o mesmo formato.
    #     - "Arrhythmia Data:" com o mesmo formato.

    #     Observação: arquivos salvos por versões anteriores desta ferramenta podem não possuir a coluna de
    #     incerteza, as âncoras ou o vínculo freq_ref; nesses casos, valores padrão são assumidos
    #     automaticamente (incerteza = uncertainty_value, âncora = valor atual, freq_ref = None,
    #     ou seja, sem vínculo — a marcação deixa de ser atualizada automaticamente caso o período seja
    #     reajustado por arraste, mas continua funcionando normalmente do contrário).

    # Parâmetros:
    #     input_file (str): Caminho para o arquivo de texto contendo os dados salvos.

    # Retorna:
    #     - infos (dict): Dicionário com os sinais dos eletrodos. Formato:
    #           {
    #             'I': {'values': [v1, v2, ...]},
    #             ...
    #           }
    #     - num_lines (int): Número de linhas (amostras) por eletrodo.
    #     - freq_data (list of tuples): Dados de períodos.
    #     - qrs_data (list of tuples): Dados de complexos QRS, cada tupla com 8 elementos
    #       (initial_x, final_x, period, duration, uncertainty, anchor_initial, anchor_final, freq_ref).
    #     - qt_data, extrasystole_data, arrhythmia_data (list of tuples): mesmo formato de qrs_data.

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

    if ecg_mono:
        header = head_mono
    else:
        header = head

    def parse_freq_ref (parts):
        if len(parts) > 7 and parts[7] != '':
            try:
                return int(parts[7])
            except ValueError:
                return None
        return None

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
                if line.split(":")[0] in header:
                    infos[line.split(":")[0]] = {'values': []}
                    ind += 1
                else:
                    y_data = line
                    infos[header[ind]]['values'].append(float(y_data))
            elif section == "Period Data":
                parts = [p.strip() for p in line.split(",")]
                initial_x, final_x, interval = parts[0], parts[1], parts[2]
                uncertainty = parts[3] if len(parts) > 3 else f"{uncertainty_value:.2f}"
                # Arquivos salvos por versões anteriores não têm âncora fixa; nesse caso a âncora
                # assume o próprio valor carregado (a faixa fica centrada na posição atual).
                anchor_initial = parts[4] if len(parts) > 4 else initial_x
                anchor_final = parts[5] if len(parts) > 5 else final_x
                freq_data.append((initial_x, final_x, interval, uncertainty, anchor_initial, anchor_final))
            elif section == "QRS Data":
                parts = [p.strip() for p in line.split(",")]
                initial_x, final_x, interval, qrs = parts[0], parts[1], parts[2], parts[3]
                uncertainty = parts[4] if len(parts) > 4 else f"{uncertainty_value:.2f}"
                anchor_initial = parts[5] if len(parts) > 5 else initial_x
                anchor_final = parts[6] if len(parts) > 6 else final_x
                freq_ref = parse_freq_ref(parts)
                # 9th field: offset-side uncertainty (tau_off for v6 marks). Files saved before
                # this existed, or hand-edited ones missing it, fall back to the onset value --
                # symmetric band either way, same as manual entries always produce.
                uncertainty_end = parts[8] if len(parts) > 8 and parts[8] != '' else uncertainty
                qrs_data.append((initial_x, final_x, interval, qrs, uncertainty, anchor_initial, anchor_final, freq_ref, uncertainty_end))
            elif section == "QT Data":
                parts = [p.strip() for p in line.split(",")]
                initial_x, final_x, interval, qt = parts[0], parts[1], parts[2], parts[3]
                uncertainty = parts[4] if len(parts) > 4 else f"{uncertainty_value:.2f}"
                anchor_initial = parts[5] if len(parts) > 5 else initial_x
                anchor_final = parts[6] if len(parts) > 6 else final_x
                freq_ref = parse_freq_ref(parts)
                qt_data.append((initial_x, final_x, interval, qt, uncertainty, anchor_initial, anchor_final, freq_ref))
            elif section == "Extrasystole Data":
                parts = [p.strip() for p in line.split(",")]
                initial_x, final_x, interval, duration = parts[0], parts[1], parts[2], parts[3]
                uncertainty = parts[4] if len(parts) > 4 else f"{uncertainty_value:.2f}"
                anchor_initial = parts[5] if len(parts) > 5 else initial_x
                anchor_final = parts[6] if len(parts) > 6 else final_x
                freq_ref = parse_freq_ref(parts)
                extrasystole_data.append((initial_x, final_x, interval, duration, uncertainty, anchor_initial, anchor_final, freq_ref))
            elif section == "Arrhythmia Data":
                parts = [p.strip() for p in line.split(",")]
                initial_x, final_x, interval, duration = parts[0], parts[1], parts[2], parts[3]
                uncertainty = parts[4] if len(parts) > 4 else f"{uncertainty_value:.2f}"
                anchor_initial = parts[5] if len(parts) > 5 else initial_x
                anchor_final = parts[6] if len(parts) > 6 else final_x
                freq_ref = parse_freq_ref(parts)
                arrhythmia_data.append((initial_x, final_x, interval, duration, uncertainty, anchor_initial, anchor_final, freq_ref))

    return infos, num_lines, freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data

def update_tables (freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data):

    # Atualiza as tabelas gráficas da interface e as listas internas da janela com os dados fornecidos.

    # Parâmetros:
    #     freq_data (list of tuples): Lista de períodos (início, fim, duração, incerteza, âncoras).
    #     qrs_data, qt_data, extrasystole_data, arrhythmia_data (list of tuples): Listas de marcações
    #     dependentes de período (início, fim, período, duração, incerteza, âncoras, freq_ref).

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

def delete_selected (event):

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
    #     - Ao remover um item de `freq_table` (Período), propaga a remoção para QRS, QT, Extrasystole e
    #       Arrhythmia via `cascade_freq_deletion`: marcações vinculadas a esse período perdem o vínculo
    #       (freq_ref = None) e marcações vinculadas a períodos posteriores têm o índice decrementado, para
    #       continuar apontando corretamente após a remoção.
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
                cascade_freq_deletion(item_index)
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

def plot_data ():

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

    # Skip non-positive durations (e.g. v6 predicting t_off < t_on, clamped
    # to 0 -- a real but invalid beat) instead of crashing on 1/0.
    estimated_normalized_velocity = []
    velocity_periods = []

    for qrs in janela.qrs:
        duration = float(qrs[3])
        if duration <= 0:
            continue
        estimated_normalized_velocity.append(1 / duration)
        velocity_periods.append(float(qrs[2]))

    if not estimated_normalized_velocity:
        message_label.config(text = "No valid QRS durations to plot.")
        plot_window.destroy()
        return

    max_value = max(estimated_normalized_velocity)

    for indice in range(len(estimated_normalized_velocity)):
        estimated_normalized_velocity[indice] /= max_value

    for indice in range(len(estimated_normalized_velocity)):
        ax_qrs.plot(velocity_periods[indice], estimated_normalized_velocity[indice], 'ro')

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

def onmotion (event):

    # Atualiza dinamicamente o valor de dx (diferença entre o ponto atual e o ponto de clique anterior)
    # enquanto o cursor se move sobre o gráfico, e também atualiza a posição da linha sendo arrastada
    # (caso o usuário esteja reajustando uma marcação dentro de sua faixa de incerteza).

    # Parâmetros:
    #     event: Evento de movimento do mouse. Deve conter `xdata` e `inaxes`.

    # Ações:
    #     - Verifica se o mouse está sobre o eixo `ax`.
    #     - Se `janela.click_state == 1`, calcula e exibe a distância horizontal (dx) entre a posição atual do mouse
    #       e o último ponto de clique armazenado em `janela.line_coords`.
    #     - Se houver uma linha de marcação sendo arrastada (`janela.dragging_line`), atualiza sua posição,
    #       restringindo-a aos limites da faixa de incerteza associada.

    if event.inaxes == ax:
        if janela.click_state == 1:
            x_atual = event.xdata
            diferenca = abs(x_atual - janela.line_coords[-1])
            dx_var.set(f"dx: {diferenca:.2f}")

    update_drag(event)

def ecg_marker():

    global janela, fig, ax, xlim, freq_table, qrs_table, qt_table, extrasystole_table, arrhythmia_table, dx_var
    global message_label, progress_bar, auto_mark_button, scrollbar, num_lines, electrodes, clean_signal, output_dir, output_file, qrs_file, qt_file
    global apd_file, vel_file, arrhythmia_file, extrasystole_file, raw_data, input_file, textbox
    global offset, ecg_mono
    global head_mono, head, head_file, uncertainty_value

    parser = argparse.ArgumentParser(description = 'ECG Marker')
    parser.add_argument('-c', required = True, dest = 'config', help = 'Path to the configuration file')
    arguments = parser.parse_args()
    config = arguments.config

    head_file, head, head_mono, input, input_file, output_dir, raw_data, clean_signal, ecg_mono, offset, uncertainty_value = read_config (config)

    # Output files
    output_file       = 'ecg_data.txt'
    qrs_file          = 'qrs_file.txt'
    qt_file           = 'qt_file.txt'
    apd_file          = 'apd_file.txt'
    vel_file          = 'vel_file.txt'
    arrhythmia_file   = 'arrhythmia_file.txt'
    extrasystole_file = 'extrasystole_file.txt'

    electrodes = {}
    num_lines = 0

    if raw_data:
        if input_file:
            electrodes, num_lines = read_file(input)
        else:
            if (ecg_mono == 0):
                electrodes, num_lines = read_dir(input)
            else:
                electrodes, num_lines = read_dir_2(input)
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

    offset_ = 0
    cont   = 0

    xlim = 1000

    fig, ax = plt.subplots()
    ax.set_xlim(0, xlim)

    for electrode in electrodes:
        if (cont != 0):
            offset_ += offset
        ax.plot(x, np.array(electrodes[electrode]['values']) + offset_, label = electrode)
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

    # Parented to `toolbar` itself (a tk.Frame under the hood), not frame_left,
    # so it appends to the same icon row as home/pan/zoom/etc. instead of
    # landing on its own row.
    ecg_nn_icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'ecg_nn.png')
    ecg_nn_icon = tk.PhotoImage(file = ecg_nn_icon_path)
    # relief=FLAT + borderwidth=0 + highlightthickness=0: no visible box around
    # the icon, matching the other (borderless) toolbar buttons.
    ecg_nn_settings_button = tk.Button(toolbar, image = ecg_nn_icon, command = open_ecg_nn_settings,
                                        relief = tk.FLAT, borderwidth = 0, highlightthickness = 0)
    ecg_nn_settings_button.image = ecg_nn_icon  # keep a reference -- PhotoImage is GC'd otherwise
    ecg_nn_settings_button.pack(side = tk.LEFT, padx = 4)

    message_label = tk.Label(janela, text = "", font = ('Arial', 12), background='white')
    message_label.grid(row = 0, column = 0, sticky = 'n', pady = 40, padx = 20)

    # Centered directly under message_label. Hidden until
    # automatic_period_marking() runs; lifted above frame_left's matplotlib
    # canvas each time it's shown so it isn't painted over.
    progress_bar = ttk.Progressbar(janela, orient = 'horizontal', mode = 'determinate', length = 250)
    progress_bar.grid(row = 0, column = 0, sticky = 'n', pady = (75, 0))
    progress_bar.grid_remove()

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

    auto_mark_button = tk.Button(frame_right, text='Automatic Marking', command = automatic_period_marking)
    auto_mark_button.grid(row = 0, column = 2, columnspan = 4, padx = 20, pady = 10, ipadx = 20)

    freq_name = tk.Label(frame_right, text = "Period", font = ('Arial', 16))
    freq_name.grid(column = 0, row = 1, columnspan = 4, padx = 2, pady = 2)

    freq_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'uncertainty'), show = 'headings', height = 5)
    freq_table.grid(row = 2, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    freq_table.heading('initial_x', text = 'Initial X')
    freq_table.heading('final_x', text = 'Final X')
    freq_table.heading('frequency', text = 'Period')
    freq_table.heading('uncertainty', text = 'Uncertainty (ms)', command = make_sort_handler('freq', 'uncertainty'))

    freq_table.column('initial_x', width = 120, anchor = 'center')
    freq_table.column('final_x', width = 120, anchor = 'center')
    freq_table.column('frequency', width = 120, anchor = 'center')
    freq_table.column('uncertainty', width = 120, anchor = 'center')

    scrollbar_vertical_freq = tk.Scrollbar(frame_right, orient='vertical', command=freq_table.yview)
    scrollbar_vertical_freq.grid(row=2, column=4, sticky='ns')
    freq_table.configure(yscrollcommand=scrollbar_vertical_freq.set)

    freq_table.bind('<<TreeviewSelect>>', freq_selected)

    qrs_name = tk.Label(frame_right, text = "QRS", font = ('Arial', 16))
    qrs_name.grid(column = 0, row = 3, columnspan = 4, padx = 2, pady = 2)

    # `columns` must match the QRS data tuple positionally (see
    # TABLE_FIELD_CONFIG['qrs']) since draw_marking_with_band and the save
    # logic read anchor_initial/anchor_final/freq_ref straight back out of
    # this widget's item values -- ttk.Treeview maps values to columns by
    # position, not by name, so the hidden fields must still occupy their
    # real slots. `displaycolumns` then picks which ones are actually shown,
    # in whatever visual order we want (here: skipping the 3 internal-only
    # anchor/freq_ref fields).
    # 'uncertainty' = onset-side (tau_on for v6); 'uncertainty_end' = offset-side (tau_off for
    # v6, symmetric duplicate of 'uncertainty' for manual entries / older saved files -- see
    # TABLE_FIELD_CONFIG['qrs'] and draw_marking_with_band).
    qrs_table = ttk.Treeview(
        frame_right,
        columns = ('initial_x', 'final_x', 'frequency', 'qrs', 'uncertainty',
                   'anchor_initial', 'anchor_final', 'freq_ref', 'uncertainty_end'),
        displaycolumns = ('initial_x', 'final_x', 'frequency', 'qrs', 'uncertainty', 'uncertainty_end'),
        show = 'headings', height = 5)
    qrs_table.grid(row = 4, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    qrs_table.heading('initial_x', text = 'Initial X')
    qrs_table.heading('final_x', text = 'Final X')
    qrs_table.heading('frequency', text = 'Period')
    qrs_table.heading('qrs', text = 'QRS')
    qrs_table.heading('uncertainty', text = 'Unc. On (ms)', command = make_sort_handler('qrs', 'uncertainty'))
    qrs_table.heading('uncertainty_end', text = 'Unc. Off (ms)')

    qrs_table.column('initial_x', width = 100, anchor = 'center')
    qrs_table.column('final_x', width = 100, anchor = 'center')
    qrs_table.column('frequency', width = 100, anchor = 'center')
    qrs_table.column('qrs', width = 100, anchor = 'center')
    qrs_table.column('uncertainty', width = 100, anchor = 'center')
    qrs_table.column('uncertainty_end', width = 100, anchor = 'center')

    scrollbar_vertical_qrs = tk.Scrollbar(frame_right, orient='vertical', command=qrs_table.yview)
    scrollbar_vertical_qrs.grid(row=4, column=4, sticky='ns')
    qrs_table.configure(yscrollcommand=scrollbar_vertical_qrs.set)

    qrs_table.bind('<<TreeviewSelect>>', qrs_selected)

    qt_name = tk.Label(frame_right, text = "QT", font = ('Arial', 16))
    qt_name.grid(column = 0, row = 5, columnspan = 4, padx = 2, pady = 2)

    qt_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'qt', 'uncertainty'), show = 'headings', height = 5)
    qt_table.grid(row = 6, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    qt_table.heading('initial_x', text = 'Initial X')
    qt_table.heading('final_x', text = 'Final X')
    qt_table.heading('frequency', text = 'Period')
    qt_table.heading('qt', text = 'QT')
    qt_table.heading('uncertainty', text = 'Uncertainty (ms)', command = make_sort_handler('qt', 'uncertainty'))

    qt_table.column('initial_x', width = 100, anchor = 'center')
    qt_table.column('final_x', width = 100, anchor = 'center')
    qt_table.column('frequency', width = 100, anchor = 'center')
    qt_table.column('qt', width = 100, anchor = 'center')
    qt_table.column('uncertainty', width = 110, anchor = 'center')

    scrollbar_vertical_qt = tk.Scrollbar(frame_right, orient='vertical', command=qt_table.yview)
    scrollbar_vertical_qt.grid(row=6, column=4, sticky='ns')
    qt_table.configure(yscrollcommand=scrollbar_vertical_qt.set)

    qt_table.bind('<<TreeviewSelect>>', qt_selected)

    extrasystole_name = tk.Label(frame_right, text = "Extrasystole", font = ('Arial', 16))
    extrasystole_name.grid(column = 0, row = 7, columnspan = 4, padx = 2, pady = 2)

    extrasystole_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'duration', 'uncertainty'), show = 'headings', height = 5)
    extrasystole_table.grid(row = 8, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    extrasystole_table.heading('initial_x', text = 'Initial X')
    extrasystole_table.heading('final_x', text = 'Final X')
    extrasystole_table.heading('frequency', text = 'Period')
    extrasystole_table.heading('duration', text = 'Duration')
    extrasystole_table.heading('uncertainty', text = 'Uncertainty (ms)', command = make_sort_handler('extrasystole', 'uncertainty'))

    extrasystole_table.column('initial_x', width = 100, anchor = 'center')
    extrasystole_table.column('final_x', width = 100, anchor = 'center')
    extrasystole_table.column('frequency', width = 100, anchor = 'center')
    extrasystole_table.column('duration', width = 100, anchor = 'center')
    extrasystole_table.column('uncertainty', width = 110, anchor = 'center')

    scrollbar_vertical_extrasystole = tk.Scrollbar(frame_right, orient='vertical', command=extrasystole_table.yview)
    scrollbar_vertical_extrasystole.grid(row=8, column=4, sticky='ns')
    extrasystole_table.configure(yscrollcommand=scrollbar_vertical_extrasystole.set)

    extrasystole_table.bind('<<TreeviewSelect>>', extrasystole_selected)

    arrhythmia_name = tk.Label(frame_right, text = "Arrhythmia", font = ('Arial', 16))
    arrhythmia_name.grid(column = 0, row = 9, columnspan = 4, padx = 2, pady = 2)

    arrhythmia_table = ttk.Treeview(frame_right, columns = ('initial_x', 'final_x', 'frequency', 'duration', 'uncertainty'), show = 'headings', height = 5)
    arrhythmia_table.grid(row = 10, column = 0, columnspan = 4, padx = 0, pady = 0, ipadx = 0, ipady = 0, sticky = 'ns')
    arrhythmia_table.heading('initial_x', text = 'Initial X')
    arrhythmia_table.heading('final_x', text = 'Final X')
    arrhythmia_table.heading('frequency', text = 'Period')
    arrhythmia_table.heading('duration', text = 'Duration')
    arrhythmia_table.heading('uncertainty', text = 'Uncertainty (ms)', command = make_sort_handler('arrhythmia', 'uncertainty'))

    arrhythmia_table.column('initial_x', width = 100, anchor = 'center')
    arrhythmia_table.column('final_x', width = 100, anchor = 'center')
    arrhythmia_table.column('frequency', width = 100, anchor = 'center')
    arrhythmia_table.column('duration', width = 100, anchor = 'center')
    arrhythmia_table.column('uncertainty', width = 110, anchor = 'center')

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

    janela.draggable_lines = {}
    janela.dragging_line = None
    janela.dragging_info = None
    janela.sort_state = {}

    dx_var = tk.StringVar()

    dx_var.set("dx: 0.00")

    dx = tk.Label(janela, textvariable=dx_var)
    dx.grid(column = 0, row = 11, padx = 10, pady = 1, ipadx = 1, ipady = 1, sticky = 'e')

    fig.canvas.mpl_connect('motion_notify_event', onmotion)
    fig.canvas.mpl_connect('button_release_event', on_release_drag)

    if (not raw_data):
        update_tables(freq_data, qrs_data, qt_data, extrasystole_data, arrhythmia_data)

    janela.bind('<Key>', key_press)

    fig.canvas.mpl_connect('button_press_event', onclick)

    janela.mainloop()

if __name__ == "__main__":
    ecg_marker()