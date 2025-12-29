import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Настройка страницы
st.set_page_config(
    page_title="Анализ и визуализация данных",
    page_icon="📊",
    layout="wide"
)

# Функция для парсинга введенных данных
def parse_data(data_text):
    """Парсит текстовые данные в массив [x, y]"""
    x_values = []
    y_values = []
    
    if not data_text:
        return np.array([]), np.array([])
    
    # Разделяем на строки
    lines = data_text.strip().split('\n')
    
    for line in lines:
        if line.strip():  # Если строка не пустая
            # Пробуем разделить по табуляции, пробелам или запятой
            parts = line.replace('\t', ' ').replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    x_values.append(x)
                    y_values.append(y)
                except ValueError:
                    continue
    
    return np.array(x_values), np.array(y_values)

# Функция для нормировки данных
def normalize_data(all_datasets, norm_type):
    """Нормирует данные в соответствии с выбранным типом"""
    if norm_type == 'Без нормировки' or not all_datasets:
        return all_datasets
    
    normalized_datasets = []
    
    if norm_type == 'Нормировка по общему максимуму':
        # Находим общий максимум среди всех данных
        global_max = 0
        for x_vals, y_vals in all_datasets:
            if len(y_vals) > 0:
                dataset_max = np.max(y_vals)
                if dataset_max > global_max:
                    global_max = dataset_max
        
        if global_max > 0:
            for x_vals, y_vals in all_datasets:
                normalized_y = y_vals / global_max
                normalized_datasets.append((x_vals, normalized_y))
        else:
            return all_datasets
    
    elif norm_type == 'Нормировка по максимуму в наборе':
        for x_vals, y_vals in all_datasets:
            if len(y_vals) > 0:
                dataset_max = np.max(y_vals)
                if dataset_max > 0:
                    normalized_y = y_vals / dataset_max
                    normalized_datasets.append((x_vals, normalized_y))
                else:
                    normalized_datasets.append((x_vals, y_vals))
    
    return normalized_datasets

# Функция для создания смещенных данных (ИСПРАВЛЕННАЯ)
def create_shifted_datasets(all_datasets, shift_offset_value):
    """Создает смещенные наборы данных для наглядного отображения"""
    shifted_datasets = []
    
    for i, (x_vals, y_vals) in enumerate(all_datasets):
        if len(y_vals) > 0:
            # Нормируем по максимуму в наборе
            dataset_max = np.max(y_vals)
            if dataset_max > 0:
                normalized_y = y_vals / dataset_max
            else:
                normalized_y = y_vals
            
            # Смещаем по Y с сохранением нулевой линии смещения
            # У каждого набора своя нулевая линия на уровне i * shift_offset_value
            base_line = i * shift_offset_value
            shifted_y = normalized_y + base_line
            shifted_datasets.append((x_vals, shifted_y, base_line))
        else:
            shifted_datasets.append((x_vals, y_vals, i * shift_offset_value))
    
    return shifted_datasets

# Функция для подготовки данных для сглаженных графиков с нулевой базой
def prepare_smooth_data_with_zero_baseline(x_vals, y_vals, smooth_sigma_value):
    """Подготавливает данные для сглаживания с добавлением крайних нулевых точек"""
    if len(x_vals) < 2 or len(y_vals) < 2:
        return x_vals, y_vals, np.array([]), np.array([])
    
    # Сортируем по X
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_vals[sorted_indices]
    
    # Добавляем крайние точки с нулевыми значениями
    x_extended = np.concatenate([[x_sorted[0] - 0.1 * (x_sorted[-1] - x_sorted[0])], 
                                  x_sorted, 
                                  [x_sorted[-1] + 0.1 * (x_sorted[-1] - x_sorted[0])]])
    y_extended = np.concatenate([[0], y_sorted, [0]])
    
    # Создаем плотную сетку для сглаженной линии
    x_dense = np.linspace(x_extended[0], x_extended[-1], 200)
    
    # Интерполяция для сглаженной линии
    f_linear = interp1d(x_extended, y_extended, kind='linear', fill_value='extrapolate')
    y_dense = f_linear(x_dense)
    
    # Применяем гауссово сглаживание
    y_smooth = gaussian_filter1d(y_dense, sigma=smooth_sigma_value)
    
    # Обрезаем сглаженную кривую до исходного диапазона X
    mask = (x_dense >= x_sorted[0]) & (x_dense <= x_sorted[-1])
    x_smooth = x_dense[mask]
    y_smooth_cropped = y_smooth[mask]
    
    return x_sorted, y_sorted, x_smooth, y_smooth_cropped

# Функция для подготовки данных для обычных сглаженных графиков
def prepare_smooth_data(x_vals, y_vals, smooth_sigma_value):
    """Подготавливает данные для обычного сглаживания"""
    if len(x_vals) < 2 or len(y_vals) < 2:
        return x_vals, y_vals, np.array([]), np.array([])
    
    # Сортируем по X
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_vals[sorted_indices]
    
    # Создаем плотную сетку для сглаженной линии
    x_dense = np.linspace(x_sorted[0], x_sorted[-1], 200)
    
    # Интерполяция для сглаженной линии
    f_linear = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    y_dense = f_linear(x_dense)
    
    # Применяем гауссово сглаживание
    y_smooth = gaussian_filter1d(y_dense, sigma=smooth_sigma_value)
    
    return x_sorted, y_sorted, x_dense, y_smooth

# Функция для определения диапазона Y
def get_y_range(all_datasets, zero_baseline=False, is_bar_chart=False):
    """Определяет диапазон Y для графиков"""
    if not all_datasets:
        return 0, 1
    
    all_y_values = []
    for dataset in all_datasets:
        if len(dataset) == 3:  # Для смещенных данных с base_line
            x_vals, y_vals, _ = dataset
        else:  # Для обычных данных
            x_vals, y_vals = dataset
            
        if len(y_vals) > 0:
            all_y_values.extend(y_vals)
    
    if not all_y_values:
        return 0, 1
    
    y_min = np.min(all_y_values)
    y_max = np.max(all_y_values)
    
    if is_bar_chart:
        # Для столбчатых диаграмм всегда от нуля
        y_min = 0
        y_range = y_max - y_min
        y_max = y_max + 0.1 * y_range
    elif zero_baseline:
        # Для сглаженных графиков с нулевой базой
        y_min = 0
        y_range = y_max - y_min
        y_max = y_max + 0.1 * y_range
    else:
        # Для обычных сглаженных графиков
        y_range = y_max - y_min
        y_min = y_min - 0.05 * y_range
        y_max = y_max + 0.1 * y_range
    
    return y_min, y_max

# Основной интерфейс Streamlit
def main():
    st.title("📊 Анализ и визуализация данных")
    st.markdown("---")
    
    # Инициализация сессионных состояний
    if 'num_datasets' not in st.session_state:
        st.session_state.num_datasets = 1
    
    if 'datasets_data' not in st.session_state:
        st.session_state.datasets_data = [""] * 10  # Максимум 10 наборов
    
    if 'dataset_names' not in st.session_state:
        st.session_state.dataset_names = [f"Набор данных {i+1}" for i in range(10)]
    
    if 'dataset_colors' not in st.session_state:
        st.session_state.dataset_colors = ['#1f77b4'] + [
            f'#{int(255*(i+1)/10):02x}{int(128*(i+1)/10):02x}{int(64*(i+1)):02x}' 
            for i in range(1, 10)
        ]
    
    if 'line_styles' not in st.session_state:
        st.session_state.line_styles = ['solid'] * 10
    
    if 'marker_styles' not in st.session_state:
        st.session_state.marker_styles = ['none'] + ['o'] * 9
    
    # Сайдбар с настройками
    with st.sidebar:
        st.header("⚙️ Настройки данных")
        
        # Управление количеством наборов данных
        num_datasets = st.slider(
            "Количество наборов данных",
            min_value=1,
            max_value=10,
            value=st.session_state.num_datasets,
            key="num_datasets_slider"
        )
        
        # Обновляем количество наборов если изменилось
        if num_datasets != st.session_state.num_datasets:
            st.session_state.num_datasets = num_datasets
            st.rerun()
        
        st.markdown("---")
        st.header("🎨 Настройки графиков")
        
        # Общие настройки
        fill_color = st.color_picker(
            "Цвет фона графиков",
            value="#ffffff",
            key="fill_color"
        )
        
        show_grid = st.checkbox(
            "Показать сетку",
            value=True,
            key="show_grid"
        )
        
        line_width = st.slider(
            "Толщина линии",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="line_width"
        )
        
        marker_size = st.slider(
            "Размер маркеров",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            key="marker_size"
        )
        
        axis_width = st.slider(
            "Толщина осей",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5,
            key="axis_width"
        )
        
        font_size = st.slider(
            "Размер шрифта заголовков",
            min_value=10,
            max_value=24,
            value=14,
            step=2,
            key="font_size"
        )
        
        x_label = st.text_input(
            "Название оси X",
            value="Ось X",
            key="x_label"
        )
        
        y_label = st.text_input(
            "Название оси Y",
            value="Ось Y",
            key="y_label"
        )
        
        graph_title = st.text_input(
            "Заголовок графиков",
            value="Анализ данных",
            key="graph_title"
        )
        
        st.markdown("---")
        st.header("📏 Настройки столбчатых диаграмм")
        
        bar_width = st.slider(
            "Ширина столбцов",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            key="bar_width"
        )
        
        bar_alpha = st.slider(
            "Прозрачность столбцов",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="bar_alpha"
        )
        
        st.markdown("---")
        st.header("🌀 Настройки сглаживания")
        
        smooth_sigma = st.slider(
            "Сила сглаживания",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="smooth_sigma"
        )
        
        smooth_zero_baseline = st.checkbox(
            "Сглаженные графики от нуля",
            value=False,
            key="smooth_zero_baseline"
        )
        
        st.markdown("---")
        st.header("📐 Настройки нормировки")
        
        normalization_type = st.selectbox(
            "Тип нормировки",
            options=['Без нормировки', 'Нормировка по общему максимуму', 'Нормировка по максимуму в наборе'],
            index=0,
            key="normalization_type"
        )
        
        st.markdown("---")
        st.header("⬆️ Настройки смещения")
        
        shift_offset = st.slider(
            "Смещение наборов",
            min_value=0.5,
            max_value=3.0,
            value=1.2,
            step=0.1,
            key="shift_offset"
        )
        
        st.markdown("---")
        st.header("🎯 Настройка диапазонов")
        
        manual_range = st.checkbox(
            "Ручная настройка диапазонов",
            value=False,
            key="manual_range"
        )
        
        if manual_range:
            col1, col2 = st.columns(2)
            with col1:
                x_min = st.number_input("X мин", value=0.0, key="x_min")
                y_min = st.number_input("Y мин", value=0.0, key="y_min")
            with col2:
                x_max = st.number_input("X макс", value=0.0, key="x_max")
                y_max = st.number_input("Y макс", value=0.0, key="y_max")
    
    # Основная область с вводом данных
    st.header("📝 Ввод данных")
    
    # Создаем табы для каждого набора данных
    tabs = st.tabs([f"Набор {i+1}" for i in range(num_datasets)])
    
    # Пустые данные по умолчанию (ИСПРАВЛЕНО)
    default_data_examples = [""] * 10
    
    for i, tab in enumerate(tabs):
        with tab:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Имя набора
                dataset_name = st.text_input(
                    f"Имя набора {i+1}",
                    value=st.session_state.dataset_names[i],
                    key=f"dataset_name_{i}"
                )
                st.session_state.dataset_names[i] = dataset_name
            
            with col2:
                # Цвет
                color = st.color_picker(
                    f"Цвет {i+1}",
                    value=st.session_state.dataset_colors[i],
                    key=f"dataset_color_{i}"
                )
                st.session_state.dataset_colors[i] = color
            
            with col3:
                # Стиль линии
                line_style = st.selectbox(
                    f"Стиль линии {i+1}",
                    options=['solid', 'dashed', 'dotted', 'dashdot'],
                    index=['solid', 'dashed', 'dotted', 'dashdot'].index(st.session_state.line_styles[i]),
                    key=f"line_style_{i}"
                )
                st.session_state.line_styles[i] = line_style
            
            with col4:
                # Маркер (ИСПРАВЛЕНО обозначение)
                marker_options = {
                    'none': 'Нет маркера',
                    'o': 'Круг',
                    's': 'Квадрат',
                    'D': 'Ромб',
                    '^': 'Треугольник вверх',
                    'v': 'Треугольник вниз',
                    'p': 'Пятиугольник',
                    '*': 'Звезда',
                    'h': 'Шестиугольник',
                    '8': 'Восьмиугольник',
                    'P': 'Заполненный плюс',
                    'X': 'Заполненный крест'
                }
                
                marker_keys = list(marker_options.keys())
                marker_labels = list(marker_options.values())
                
                # Определяем текущий индекс
                current_marker = st.session_state.marker_styles[i]
                current_index = marker_keys.index(current_marker) if current_marker in marker_keys else 0
                
                marker_style_label = st.selectbox(
                    f"Маркер {i+1}",
                    options=marker_labels,
                    index=current_index,
                    key=f"marker_style_label_{i}"
                )
                
                # Сохраняем ключ маркера
                selected_index = marker_labels.index(marker_style_label)
                st.session_state.marker_styles[i] = marker_keys[selected_index]
            
            # Поле для ввода данных (ИСПРАВЛЕНО - пустое по умолчанию)
            data_text = st.text_area(
                f"Данные набора {i+1} (формат: X Y в каждой строке)",
                value=st.session_state.datasets_data[i],
                height=150,
                key=f"data_text_{i}",
                placeholder="Введите данные в формате:\n10.0 20.5\n15.0 30.2\n20.0 25.7\n\nИли:\n10.0\t20.5\n15.0\t30.2\n20.0\t25.7"
            )
            st.session_state.datasets_data[i] = data_text
    
    # Кнопка для обновления графиков
    if st.button("🔄 Обновить графики", type="primary", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    
    # Сбор всех наборов данных
    all_datasets = []
    all_names = []
    all_colors = []
    all_line_styles = []
    all_marker_styles = []
    
    for i in range(num_datasets):
        data_text = st.session_state.datasets_data[i]
        x_vals, y_vals = parse_data(data_text)
        
        if len(x_vals) > 0 and len(y_vals) > 0:
            all_datasets.append((x_vals, y_vals))
            all_names.append(st.session_state.dataset_names[i])
            all_colors.append(st.session_state.dataset_colors[i])
            all_line_styles.append(st.session_state.line_styles[i])
            all_marker_styles.append(st.session_state.marker_styles[i])
    
    if not all_datasets:
        st.warning("❌ Нет данных для построения графиков. Пожалуйста, введите данные.")
        return
    
    # Получаем нормированные и смещенные данные
    norm_datasets = normalize_data(all_datasets, normalization_type)
    shifted_datasets_with_base = create_shifted_datasets(all_datasets, shift_offset)
    
    # Разделяем смещенные данные на данные и базовые линии
    shifted_datasets = [(x_vals, y_vals) for x_vals, y_vals, _ in shifted_datasets_with_base]
    base_lines = [base_line for _, _, base_line in shifted_datasets_with_base]
    
    # Определяем общий диапазон X
    all_x_values = []
    for x_vals, y_vals in all_datasets:
        if len(x_vals) > 0:
            all_x_values.extend(x_vals)
    
    if len(all_x_values) > 0:
        if manual_range and st.session_state.get('x_min', 0) != st.session_state.get('x_max', 0):
            x_min = st.session_state.x_min
            x_max = st.session_state.x_max
        else:
            x_range = max(all_x_values) - min(all_x_values)
            x_min = min(all_x_values) - 0.1 * x_range
            x_max = max(all_x_values) + 0.1 * x_range
    else:
        x_min, x_max = 0, 1
    
    # Создаем графики
    st.header("📈 Графики")
    
    # Создаем 6 графиков в виде колонок
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    titles = [
        'Столбчатая диаграмма X-Y',
        'Сглаженный график с заливкой',
        'Нормированная столбчатая диаграмма',
        'Нормированный сглаженный график',
        'Смещенные нормированные столбцы',
        'Смещенные нормированные кривые'
    ]
    
    # Установка фона для всех графиков
    for ax in axes:
        ax.set_facecolor(fill_color)
    
    # График 1: Столбчатая диаграмма X-Y
    for idx, (x_vals, y_vals) in enumerate(all_datasets):
        if len(x_vals) > 0 and len(y_vals) > 0:
            sorted_indices = np.argsort(x_vals)
            x_sorted = x_vals[sorted_indices]
            y_sorted = y_vals[sorted_indices]
            
            if len(x_sorted) > 1:
                width = (x_sorted[-1] - x_sorted[0]) / (len(x_sorted) * 1.5) * bar_width
            else:
                width = bar_width
            
            ax1.bar(x_sorted, y_sorted, 
                   width=width,
                   alpha=bar_alpha, 
                   color=all_colors[idx],
                   edgecolor='black', 
                   linewidth=1,
                   label=all_names[idx])
    
    y_min_1, y_max_1 = get_y_range(all_datasets, is_bar_chart=True)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax1.set_ylim(st.session_state.y_min, st.session_state.y_max)
    else:
        ax1.set_ylim(y_min_1, y_max_1)
    
    ax1.set_xlabel('Позиция (X)', fontsize=font_size)
    ax1.set_ylabel('Высота (Y)', fontsize=font_size)
    ax1.set_title(titles[0], fontsize=font_size + 2, pad=20)
    ax1.set_xlim(x_min, x_max)
    if len(all_datasets) > 1:
        ax1.legend(fontsize=font_size - 2, loc='best')
    
    # График 2: Сглаженный график с заливкой
    for idx, (x_vals, y_vals) in enumerate(all_datasets):
        if len(x_vals) > 3 and len(y_vals) > 3:
            if smooth_zero_baseline:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data_with_zero_baseline(
                    x_vals, y_vals, smooth_sigma)
            else:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data(
                    x_vals, y_vals, smooth_sigma)
            
            if len(x_smooth) > 0 and len(y_smooth) > 0:
                ax2.plot(x_smooth, y_smooth,
                        linewidth=line_width + 1,
                        color=all_colors[idx],
                        linestyle=all_line_styles[idx],
                        alpha=0.9,
                        label=all_names[idx] if len(all_datasets) == 1 else None)
                
                ax2.fill_between(x_smooth, y_smooth, alpha=0.2, color=all_colors[idx])
                
                if all_marker_styles[idx] != 'none':
                    ax2.scatter(x_sorted, y_sorted,
                              s=marker_size,
                              c=all_colors[idx],
                              marker=all_marker_styles[idx],
                              alpha=0.7,
                              edgecolors='k',
                              linewidths=0.5,
                              zorder=5)
    
    y_min_2, y_max_2 = get_y_range(all_datasets, smooth_zero_baseline)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax2.set_ylim(st.session_state.y_min, st.session_state.y_max)
    else:
        ax2.set_ylim(y_min_2, y_max_2)
    
    ax2.set_xlabel(x_label, fontsize=font_size)
    ax2.set_ylabel(y_label, fontsize=font_size)
    ax2.set_title(titles[1], fontsize=font_size + 2, pad=20)
    ax2.set_xlim(x_min, x_max)
    if len(all_datasets) == 1:
        ax2.legend(fontsize=font_size - 2, loc='best')
    
    # График 3: Нормированная столбчатая диаграмма
    for idx, (x_vals, y_vals) in enumerate(norm_datasets):
        if len(x_vals) > 0 and len(y_vals) > 0:
            sorted_indices = np.argsort(x_vals)
            x_sorted = x_vals[sorted_indices]
            y_sorted = y_vals[sorted_indices]
            
            if len(x_sorted) > 1:
                width = (x_sorted[-1] - x_sorted[0]) / (len(x_sorted) * 1.5) * bar_width
            else:
                width = bar_width
            
            ax3.bar(x_sorted, y_sorted, 
                   width=width,
                   alpha=bar_alpha, 
                   color=all_colors[idx],
                   edgecolor='black', 
                   linewidth=1,
                   label=all_names[idx])
    
    y_min_3, y_max_3 = get_y_range(norm_datasets, is_bar_chart=True)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax3.set_ylim(st.session_state.y_min, st.session_state.y_max)
    elif normalization_type != 'Без нормировки':
        ax3.set_ylim(0, 1.2)
    else:
        ax3.set_ylim(y_min_3, y_max_3)
    
    ax3.set_xlabel('Позиция (X)', fontsize=font_size)
    ax3.set_ylabel('Нормированная высота', fontsize=font_size)
    ax3.set_title(titles[2], fontsize=font_size + 2, pad=20)
    ax3.set_xlim(x_min, x_max)
    if len(norm_datasets) > 1:
        ax3.legend(fontsize=font_size - 2, loc='best')
    
    # График 4: Нормированный сглаженный график
    for idx, (x_vals, y_vals) in enumerate(norm_datasets):
        if len(x_vals) > 3 and len(y_vals) > 3:
            if smooth_zero_baseline:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data_with_zero_baseline(
                    x_vals, y_vals, smooth_sigma)
            else:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data(
                    x_vals, y_vals, smooth_sigma)
            
            if len(x_smooth) > 0 and len(y_smooth) > 0:
                ax4.plot(x_smooth, y_smooth,
                        linewidth=line_width + 1,
                        color=all_colors[idx],
                        linestyle=all_line_styles[idx],
                        alpha=0.9,
                        label=all_names[idx] if len(norm_datasets) == 1 else None)
                
                ax4.fill_between(x_smooth, y_smooth, alpha=0.2, color=all_colors[idx])
                
                if all_marker_styles[idx] != 'none':
                    ax4.scatter(x_sorted, y_sorted,
                              s=marker_size,
                              c=all_colors[idx],
                              marker=all_marker_styles[idx],
                              alpha=0.7,
                              edgecolors='k',
                              linewidths=0.5,
                              zorder=5)
    
    y_min_4, y_max_4 = get_y_range(norm_datasets, smooth_zero_baseline)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax4.set_ylim(st.session_state.y_min, st.session_state.y_max)
    else:
        ax4.set_ylim(y_min_4, y_max_4)
    
    ax4.set_xlabel(x_label, fontsize=font_size)
    ax4.set_ylabel('Нормированное значение', fontsize=font_size)
    ax4.set_title(titles[3], fontsize=font_size + 2, pad=20)
    ax4.set_xlim(x_min, x_max)
    if len(norm_datasets) == 1:
        ax4.legend(fontsize=font_size - 2, loc='best')
    
    # График 5: Смещенные нормированные столбцы (ИСПРАВЛЕННЫЙ)
    for idx, (x_vals, y_vals) in enumerate(shifted_datasets):
        if len(x_vals) > 0 and len(y_vals) > 0:
            sorted_indices = np.argsort(x_vals)
            x_sorted = x_vals[sorted_indices]
            y_sorted = y_vals[sorted_indices]
            
            if len(x_sorted) > 1:
                width = (x_sorted[-1] - x_sorted[0]) / (len(x_sorted) * 1.5) * bar_width
            else:
                width = bar_width
            
            # Рисуем столбцы от базовой линии
            ax5.bar(x_sorted, y_sorted - base_lines[idx], 
                   width=width,
                   bottom=base_lines[idx],  # Указываем базовую линию
                   alpha=bar_alpha, 
                   color=all_colors[idx],
                   edgecolor='black', 
                   linewidth=1,
                   label=all_names[idx])
            
            # Рисуем базовую линию для наглядности
            ax5.axhline(y=base_lines[idx], color=all_colors[idx], linestyle='--', alpha=0.5, linewidth=1)
    
    y_min_5, y_max_5 = get_y_range(shifted_datasets, is_bar_chart=False)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax5.set_ylim(st.session_state.y_min, st.session_state.y_max)
    else:
        ax5.set_ylim(y_min_5, y_max_5)
    
    ax5.set_xlabel('Позиция (X)', fontsize=font_size)
    ax5.set_ylabel('Смещенные нормированные значения', fontsize=font_size)
    ax5.set_title(titles[4], fontsize=font_size + 2, pad=20)
    ax5.set_xlim(x_min, x_max)
    if len(shifted_datasets) > 0:
        ax5.legend(fontsize=font_size - 2, loc='best')
    
    # График 6: Смещенные нормированные кривые (ИСПРАВЛЕННЫЙ)
    for idx, (x_vals, y_vals) in enumerate(shifted_datasets):
        if len(x_vals) > 3 and len(y_vals) > 3:
            if smooth_zero_baseline:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data_with_zero_baseline(
                    x_vals, y_vals, smooth_sigma)
            else:
                x_sorted, y_sorted, x_smooth, y_smooth = prepare_smooth_data(
                    x_vals, y_vals, smooth_sigma)
            
            if len(x_smooth) > 0 and len(y_smooth) > 0:
                ax6.plot(x_smooth, y_smooth,
                        linewidth=line_width + 1,
                        color=all_colors[idx],
                        linestyle=all_line_styles[idx],
                        alpha=0.9,
                        label=all_names[idx] if len(shifted_datasets) == 1 else None)
                
                # Заполнение от базовой линии
                ax6.fill_between(x_smooth, base_lines[idx], y_smooth, alpha=0.2, color=all_colors[idx])
                
                # Рисуем базовую линию для наглядности
                ax6.axhline(y=base_lines[idx], color=all_colors[idx], linestyle='--', alpha=0.5, linewidth=1)
                
                if all_marker_styles[idx] != 'none':
                    ax6.scatter(x_sorted, y_sorted,
                              s=marker_size,
                              c=all_colors[idx],
                              marker=all_marker_styles[idx],
                              alpha=0.7,
                              edgecolors='k',
                              linewidths=0.5,
                              zorder=5)
    
    y_min_6, y_max_6 = get_y_range(shifted_datasets, smooth_zero_baseline)
    if manual_range and st.session_state.get('y_min', 0) != st.session_state.get('y_max', 0):
        ax6.set_ylim(st.session_state.y_min, st.session_state.y_max)
    else:
        ax6.set_ylim(y_min_6, y_max_6)
    
    ax6.set_xlabel(x_label, fontsize=font_size)
    ax6.set_ylabel('Смещенные нормированные значения', fontsize=font_size)
    ax6.set_title(titles[5], fontsize=font_size + 2, pad=20)
    ax6.set_xlim(x_min, x_max)
    if len(shifted_datasets) > 0:
        ax6.legend(fontsize=font_size - 2, loc='best')
    
    # Общие настройки для всех графиков
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(axis_width)
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.grid(False)
        
        ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
    
    # Отображаем графики в Streamlit
    cols = st.columns(2)
    
    with cols[0]:
        st.pyplot(fig1)
        st.pyplot(fig3)
        st.pyplot(fig5)
    
    with cols[1]:
        st.pyplot(fig2)
        st.pyplot(fig4)
        st.pyplot(fig6)
    
    # Информация о графиках
    with st.expander("📋 Информация о графиках"):
        st.markdown("""
        ### 6 типов графиков:
        
        1. **Столбчатая диаграмма X-Y** - отображает исходные данные в виде столбцов
        2. **Сглаженный график с заливкой** - показывает сглаженную кривую с заливкой под ней
        3. **Нормированная столбчатая диаграмма** - столбцы с нормированными значениями
        4. **Нормированный сглаженный график** - сглаженная кривая с нормированными значениями
        5. **Смещенные нормированные столбцы** - нормированные данные со смещением по Y для разделения наборов
        6. **Смещенные нормированные кривые** - сглаженные кривые со смещением по Y
        
        ### Особенности:
        - **Сглаженные графики от нуля**: при включении добавляются крайние точки с нулевыми значениями
        - **Нормировка**: можно нормировать по общему максимуму или по максимуму в каждом наборе
        - **Смещение**: позволяет визуально разделить несколько наборов данных
        - **Базовые линии**: на смещенных графиках показаны базовые линии каждого набора
        
        ### Формат ввода данных:
        - Каждая строка должна содержать два числа: X и Y
        - Разделитель: пробел, табуляция или запятая
        - Пример: `10.0 20.5` или `10.0\t20.5` или `10.0, 20.5`
        """)

if __name__ == "__main__":
    main()
