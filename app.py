"""
Web-based Image Processor with OpenCV
=====================================

Este projeto é um processador de imagens baseado na web que utiliza Flask e OpenCV 
para aplicar diversos filtros e técnicas de processamento de imagem.

Funcionalidades principais:
- Segmentação adaptativa
- Detecção de bordas (Canny, Sobel)
- Filtros de suavização (Gaussian, Median)
- Equalização de histograma
- Limiarização (Otsu, Manual)
- Análise de histograma

Autor: Desenvolvido para processamento científico de imagens
Data: 2025
"""

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar logging para monitoramento
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configurações da aplicação
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', './Entrada')
app.config['PROCESSED_FOLDER'] = os.getenv('PROCESSED_FOLDER', './Saida')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 16)) * 1024 * 1024  # MB em bytes

# Criar diretórios se não existirem
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

class ImageProcessorFilters:
    """
    Classe responsável por aplicar diversos filtros de processamento de imagem
    usando OpenCV e outras bibliotecas especializadas.
    
    Esta classe implementa algoritmos fundamentais de visão computacional:
    - Filtros de suavização e redução de ruído
    - Detecção de bordas e contornos
    - Segmentação e limiarização
    - Análise estatística de imagens
    """
    
    def __init__(self):
        """
        Inicializa a classe com os diretórios de entrada e saída.
        """
        self.output_folder = app.config['PROCESSED_FOLDER']
        self.input_folder = app.config['UPLOAD_FOLDER']
        logger.info("ImageProcessorFilters inicializado com sucesso")

    def _generate_output_filename(self, original_name: str, filter_name: str) -> str:
        """
        Gera um nome de arquivo de saída baseado no nome original e no filtro aplicado.
        
        Args:
            original_name (str): Nome do arquivo original
            filter_name (str): Nome do filtro aplicado
            
        Returns:
            str: Caminho completo do arquivo de saída
        """
        name_parts = original_name.rsplit('.', 1)
        if len(name_parts) == 2:
            new_name = f"{name_parts[0]}_{filter_name}.{name_parts[1]}"
        else:
            new_name = f"{original_name}_{filter_name}"
        
        return os.path.join(self.output_folder, new_name)

    def adaptive_threshold_segmentation(self, img: np.ndarray, filename: str, kernel_size: int) -> str:
        """
        Aplica segmentação por limiarização adaptativa.
        
        A limiarização adaptativa é útil quando a imagem tem condições de iluminação variáveis.
        O algoritmo calcula o limiar para cada pixel baseado nos pixels vizinhos.
        
        Processo:
        1. Converte a imagem para escala de cinza
        2. Aplica filtro mediano para reduzir ruído
        3. Calcula limiar adaptativo usando a média local
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            kernel_size (int): Tamanho do kernel para filtro mediano (deve ser ímpar)
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtro mediano para reduzir ruído salt-and-pepper
            smoothed = cv2.medianBlur(gray_img, kernel_size)
            
            # Aplicar limiarização adaptativa
            adaptive_thresh = cv2.adaptiveThreshold(
                smoothed, 
                255,                           # Valor máximo atribuído
                cv2.ADAPTIVE_THRESH_MEAN_C,    # Método adaptativo baseado na média
                cv2.THRESH_BINARY,             # Tipo de limiarização binária
                11,                            # Tamanho da área de vizinhança
                2                              # Constante subtraída da média
            )
            
            output_path = self._generate_output_filename(filename, "AdaptiveThreshold")
            cv2.imwrite(output_path, adaptive_thresh)
            
            logger.info(f"Segmentação adaptativa aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na segmentação adaptativa: {str(e)}")
            raise

    def canny_edge_detection(self, img: np.ndarray, filename: str, blur_kernel: int, low_threshold: int, high_threshold: int) -> str:
        """
        Aplica detecção de bordas usando o algoritmo Canny.
        
        O detector de bordas Canny é um dos mais populares e eficazes algoritmos de detecção de bordas.
        
        Processo do algoritmo Canny:
        1. Suavização com filtro Gaussiano para reduzir ruído
        2. Cálculo do gradiente de intensidade da imagem
        3. Supressão de não-máximos para afinar as bordas
        4. Limiarização por histerese com dois limiares
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            blur_kernel (int): Tamanho do kernel Gaussiano (deve ser ímpar)
            low_threshold (int): Limiar inferior para histerese (0-255)
            high_threshold (int): Limiar superior para histerese (0-255)
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar suavização Gaussiana para reduzir ruído
            blurred = cv2.GaussianBlur(gray_img, (blur_kernel, blur_kernel), 0)
            
            # Aplicar detecção de bordas Canny
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            
            output_path = self._generate_output_filename(filename, "CannyEdges")
            cv2.imwrite(output_path, edges)
            
            logger.info(f"Detecção Canny aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na detecção Canny: {str(e)}")
            raise

    def object_counting_with_contours(self, img: np.ndarray, filename: str) -> str:
        """
        Detecta e conta objetos na imagem usando contornos.
        
        Este método é útil para análise quantitativa de imagens, especialmente
        em aplicações científicas como contagem de células, partículas, etc.
        
        Processo:
        1. Conversão para escala de cinza
        2. Limiarização binária invertida
        3. Operação morfológica de dilatação para conectar objetos próximos
        4. Detecção de contornos externos
        5. Contagem e anotação dos objetos encontrados
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            
        Returns:
            str: Caminho do arquivo processado com objetos contados
            
        Note:
            Este método funciona melhor com objetos bem definidos e contrastados.
        """
        try:
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Limiarização binária invertida
            _, binary_thresh = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)
            
            # Elemento estruturante para operações morfológicas
            kernel = np.ones((2, 2), np.uint8)
            
            # Dilatação para conectar partes próximas dos objetos
            dilated = cv2.dilate(binary_thresh, kernel, iterations=2)
            
            # Encontrar contornos externos
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Contar objetos
            object_count = len(contours)
            
            # Adicionar texto com contagem na imagem
            count_text = f"Objetos detectados: {object_count}"
            cv2.putText(
                dilated, count_text, 
                (10, 30),                    # Posição do texto
                cv2.FONT_HERSHEY_SIMPLEX,    # Fonte
                0.7,                         # Escala da fonte
                (255, 255, 255),             # Cor (branco)
                2                            # Espessura
            )
            
            output_path = self._generate_output_filename(filename, "ObjectCounting")
            cv2.imwrite(output_path, dilated)
            
            logger.info(f"Contagem de objetos concluída: {object_count} objetos em {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na contagem de objetos: {str(e)}")
            raise
    
    def histogram_equalization(self, img: np.ndarray, filename: str) -> str:
        """
        Aplica equalização de histograma para melhorar o contraste da imagem.
        
        A equalização de histograma é uma técnica que redistribui os valores de intensidade
        para utilizar toda a gama dinâmica disponível, melhorando o contraste global.
        
        Para imagens coloridas, a equalização é aplicada no canal de luminância (Y)
        do espaço de cor YUV para preservar as informações de cor.
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Converter para espaço de cor YUV
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            # Equalizar apenas o canal de luminância (Y)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            
            # Converter de volta para BGR
            equalized_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            output_path = self._generate_output_filename(filename, "HistogramEqualized")
            cv2.imwrite(output_path, equalized_img)
            
            logger.info(f"Equalização de histograma aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na equalização de histograma: {str(e)}")
            raise

    def gaussian_blur(self, img: np.ndarray, filename: str, sigma: int, kernel_size: int) -> str:
        """
        Aplica filtro Gaussiano para suavização da imagem.
        
        O filtro Gaussiano é um filtro de suavização linear que utiliza a função
        Gaussiana para calcular pesos. É muito eficaz para redução de ruído
        preservando bordas melhor que filtros de média simples.
        
        A função Gaussiana 2D é definida por:
        G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            sigma (int): Desvio padrão da função Gaussiana (controla a intensidade da suavização)
            kernel_size (int): Tamanho do kernel (deve ser ímpar e positivo)
            
        Returns:
            str: Caminho do arquivo processado
            
        Note:
            Valores maiores de sigma resultam em maior suavização
        """
        try:
            # Garantir que o kernel_size é ímpar
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Aplicar filtro Gaussiano
            blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            
            output_path = self._generate_output_filename(filename, "GaussianBlur")
            cv2.imwrite(output_path, blurred_img)
            
            logger.info(f"Filtro Gaussiano aplicado com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro no filtro Gaussiano: {str(e)}")
            raise

    def generate_histogram_plot(self, img: np.ndarray, filename: str) -> str:
        """
        Gera e salva o histograma da imagem.
        
        O histograma é uma representação gráfica da distribuição de intensidades
        dos pixels na imagem. É fundamental para análise de contraste, brilho
        e características estatísticas da imagem.
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            
        Returns:
            str: Caminho do arquivo do histograma gerado
        """
        try:
            # Converter para escala de cinza se necessário
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
            
            # Calcular histograma
            hist, bin_edges = np.histogram(gray_img.ravel(), 256, [0, 256])
            
            # Configurar matplotlib para não usar interface gráfica
            plt.figure(figsize=(10, 6))
            plt.plot(bin_edges[0:-1], hist, color='black', linewidth=2)
            plt.xlabel('Intensidade de Pixel')
            plt.ylabel('Frequência')
            plt.title('Histograma da Imagem')
            plt.xlim([0, 255])
            plt.grid(True, alpha=0.3)
            
            # Salvar histograma como PNG
            output_path = self._generate_output_filename(filename.rsplit('.', 1)[0] + '.png', "Histogram")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Histograma gerado com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na geração do histograma: {str(e)}")
            raise

    def manual_threshold_segmentation(self, img: np.ndarray, filename: str, kernel_size: int, low_thresh: int, high_thresh: int) -> str:
        """
        Aplica segmentação por limiarização manual.
        
        A limiarização manual permite controle direto sobre os valores de limiar,
        útil quando se conhece as características específicas da imagem.
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            kernel_size (int): Tamanho do kernel para filtro mediano
            low_thresh (int): Valor de limiar inferior (0-255)
            high_thresh (int): Valor de limiar superior (0-255)
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Aplicar filtro mediano para reduzir ruído
            smoothed = cv2.medianBlur(img, kernel_size)
            
            # Aplicar limiarização binária
            _, binary_img = cv2.threshold(smoothed, low_thresh, high_thresh, cv2.THRESH_BINARY)
            
            output_path = self._generate_output_filename(filename, "ManualThreshold")
            cv2.imwrite(output_path, binary_img)
            
            logger.info(f"Limiarização manual aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na limiarização manual: {str(e)}")
            raise

    def mean_blur_filter(self, img: np.ndarray, filename: str, kernel_size: int) -> str:
        """
        Aplica filtro de média (mean blur) para suavização.
        
        O filtro de média substitui cada pixel pela média aritmética dos pixels
        em sua vizinhança. É útil para redução de ruído, mas pode borrar bordas.
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            kernel_size (int): Tamanho do kernel quadrado
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Aplicar filtro de média
            blurred_img = cv2.blur(img, (kernel_size, kernel_size))
            
            output_path = self._generate_output_filename(filename, "MeanBlur")
            cv2.imwrite(output_path, blurred_img)
            
            logger.info(f"Filtro de média aplicado com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro no filtro de média: {str(e)}")
            raise

    def otsu_threshold(self, img: np.ndarray, filename: str) -> str:
        """
        Aplica limiarização automática usando o método de Otsu.
        
        O método de Otsu determina automaticamente o valor de limiar ótimo
        maximizando a variância entre classes (fundo e objeto).
        É muito eficaz para imagens com histograma bimodal.
        
        Vantagens do método de Otsu:
        - Não requer definição manual de limiar
        - Minimiza a variância intra-classe
        - Maximiza a variância inter-classe
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar suavização Gaussiana
            smoothed = cv2.GaussianBlur(gray_img, (5, 5), 0)
            
            # Aplicar limiarização de Otsu
            threshold_value, otsu_img = cv2.threshold(
                smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            logger.info(f"Valor de limiar Otsu calculado: {threshold_value}")
            
            output_path = self._generate_output_filename(filename, "OtsuThreshold")
            cv2.imwrite(output_path, otsu_img)
            
            logger.info(f"Limiarização Otsu aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na limiarização Otsu: {str(e)}")
            raise

    def sobel_edge_detection(self, img: np.ndarray, filename: str, dx: int, dy: int, kernel_size: int) -> str:
        """
        Aplica detecção de bordas usando o operador Sobel.
        
        O operador Sobel calcula o gradiente da intensidade da imagem usando
        kernels de convolução 3x3. É particularmente útil para detectar bordas
        em direções específicas (horizontal, vertical ou ambas).
        
        Kernels Sobel:
        Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
              [-2, 0, 2],           [ 0,  0,  0],
              [-1, 0, 1]]           [ 1,  2,  1]]
        
        Args:
            img (np.ndarray): Imagem de entrada
            filename (str): Nome do arquivo
            dx (int): Ordem da derivada em x (0 ou 1)
            dy (int): Ordem da derivada em y (0 ou 1)
            kernel_size (int): Tamanho do kernel Sobel (deve ser ímpar)
            
        Returns:
            str: Caminho do arquivo processado
        """
        try:
            # Converter para escala de cinza
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar suavização Gaussiana
            smoothed = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
            
            # Calcular gradientes Sobel
            sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, dx, 0, ksize=kernel_size)
            sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, dy, ksize=kernel_size)
            
            # Converter para uint8 e calcular magnitude
            sobel_x = np.uint8(np.absolute(sobel_x))
            sobel_y = np.uint8(np.absolute(sobel_y))
            
            # Combinar gradientes usando OR bitwise
            combined_sobel = cv2.bitwise_or(sobel_x, sobel_y)
            
            output_path = self._generate_output_filename(filename, "SobelEdges")
            cv2.imwrite(output_path, combined_sobel)
            
            logger.info(f"Detecção Sobel aplicada com sucesso: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro na detecção Sobel: {str(e)}")
            raise


# Instância global da classe de filtros
image_processor = ImageProcessorFilters()

"""
=== ROTAS DA API FLASK ===
Cada rota corresponde a um filtro específico de processamento de imagem.
"""

@app.route('/')
def home():
    """
    Rota principal que serve a página HTML do aplicativo.
    
    Returns:
        str: Template HTML renderizado
    """
    return render_template('index.html')

@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    """
    Serve imagens processadas do diretório de saída.
    
    Args:
        filename (str): Nome do arquivo de imagem
        
    Returns:
        Response: Arquivo de imagem
    """
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

def save_uploaded_file(file) -> tuple:
    """
    Salva arquivo enviado pelo usuário e retorna informações necessárias.
    
    Args:
        file: Arquivo enviado via formulário
        
    Returns:
        tuple: (filepath, filename, loaded_image)
    """
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    img = cv2.imread(filepath)
    return filepath, filename, img

@app.route('/adaptativeSegmentation', methods=['POST'])
def adaptive_segmentation_route():
    """
    Endpoint para aplicar segmentação adaptativa.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - mediana: Tamanho do kernel para filtro mediano
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        kernel_size = int(request.form['mediana'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.adaptive_threshold_segmentation(img, filename, kernel_size)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de segmentação adaptativa: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/canny', methods=['POST'])
def canny_route():
    """
    Endpoint para detecção de bordas Canny.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - mediana: Tamanho do kernel para suavização
    - canny1: Limiar inferior
    - canny2: Limiar superior
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        blur_kernel = int(request.form['mediana'])
        low_threshold = int(request.form['canny1'])
        high_threshold = int(request.form['canny2'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.canny_edge_detection(img, filename, blur_kernel, low_threshold, high_threshold)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota Canny: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/cannyEdge', methods=['POST'])
def canny_edge_route():
    """
    Endpoint para contagem de objetos com contornos.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.object_counting_with_contours(img, filename)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de contagem de objetos: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/equalize', methods=['POST'])
def equalize_route():
    """
    Endpoint para equalização de histograma.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.histogram_equalization(img, filename)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de equalização: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/gaussian', methods=['POST'])
def gaussian_route():
    """
    Endpoint para filtro Gaussiano.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - sigma: Desvio padrão
    - ksize: Tamanho do kernel
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        sigma = int(request.form['sigma'])
        kernel_size = int(request.form['ksize'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.gaussian_blur(img, filename, sigma, kernel_size)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota Gaussiana: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/histogram', methods=['POST'])
def histogram_route():
    """
    Endpoint para geração de histograma.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.generate_histogram_plot(img, filename)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de histograma: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/manualSegmentation', methods=['POST'])
def manual_segmentation_route():
    """
    Endpoint para segmentação manual.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - mediana: Tamanho do kernel
    - threshold1Number: Limiar inferior
    - threshold2Number: Limiar superior
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        kernel_size = int(request.form['mediana'])
        thresh1 = int(request.form['threshold1Number'])
        thresh2 = int(request.form['threshold2Number'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.manual_threshold_segmentation(img, filename, kernel_size, thresh1, thresh2)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de segmentação manual: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/mediam', methods=['POST'])
def mean_blur_route():
    """
    Endpoint para filtro de média.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - mediana: Tamanho do kernel
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        kernel_size = int(request.form['mediana'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.mean_blur_filter(img, filename, kernel_size)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota de filtro de média: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/otsu', methods=['POST'])
def otsu_route():
    """
    Endpoint para limiarização Otsu.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.otsu_threshold(img, filename)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota Otsu: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/sobel', methods=['POST'])
def sobel_route():
    """
    Endpoint para detecção de bordas Sobel.
    
    Parâmetros esperados:
    - file: Arquivo de imagem
    - dx: Derivada em x
    - dy: Derivada em y
    - ksize: Tamanho do kernel
    
    Returns:
        JSON: Caminho de saída e URL da imagem processada
    """
    try:
        file = request.files['file']
        dx = int(request.form['dx'])
        dy = int(request.form['dy'])
        kernel_size = int(request.form['ksize'])
        
        filepath, filename, img = save_uploaded_file(file)
        
        output_path = image_processor.sobel_edge_detection(img, filename, dx, dy, kernel_size)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
        
    except Exception as e:
        logger.error(f"Erro na rota Sobel: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


"""
=== CONFIGURAÇÃO DE EXECUÇÃO ===
"""

if __name__ == '__main__':
    # Configurações de desenvolvimento/produção baseadas em variáveis de ambiente
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 7000))
    
    logger.info(f"Iniciando aplicação Flask em {host}:{port} (Debug: {debug_mode})")
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        threaded=True  # Permite múltiplas requisições simultâneas
    )

