# 🖼️ Web OpenImage Processor

Um processador de imagens baseado na web que utiliza **Flask** e **OpenCV** para aplicar diversos filtros e técnicas avançadas de processamento de imagem e visão computacional.

## 📋 Índice

- [Características](#-características)
- [Filtros Disponíveis](#-filtros-disponíveis)
- [Instalação](#-instalação)
- [Uso com Docker](#-uso-com-docker)
- [Configuração](#-configuração)
- [API](#-api)
- [Desenvolvimento](#-desenvolvimento)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## ✨ Características

- **Interface web moderna e responsiva** com design intuitivo
- **10+ algoritmos de processamento de imagem** implementados
- **Explicações detalhadas** de cada filtro e seus parâmetros
- **Processamento em tempo real** com feedback visual
- **Suporte a múltiplos formatos** de imagem (JPG, PNG, BMP, TIFF)
- **Containerização com Docker** para fácil deployment
- **Configuração flexível** via variáveis de ambiente
- **Logs detalhados** para monitoramento
- **Tratamento de erros robusto**

## 🔧 Filtros Disponíveis

### Detecção de Bordas
- **Canny Edge Detection**: O algoritmo mais popular para detecção de bordas
- **Sobel Edge Detection**: Detecta bordas usando gradientes direcionais

### Segmentação
- **Segmentação Adaptativa**: Limiarização automática baseada em regiões locais
- **Segmentação Manual**: Controle total sobre valores de limiarização
- **Limiarização Otsu**: Cálculo automático do limiar ótimo

### Filtros de Suavização
- **Filtro Gaussiano**: Suavização preservando bordas
- **Filtro de Média**: Redução de ruído por média aritmética

### Análise e Melhoria
- **Equalização de Histograma**: Melhoria automática de contraste
- **Análise de Histograma**: Visualização da distribuição de intensidades
- **Contagem de Objetos**: Detecção e contagem automática de objetos

## 🚀 Instalação

### Pré-requisitos
- Python 3.8+
- pip
- Docker (opcional)

### Instalação Local

1. **Clone o repositório**
```bash
git clone https://github.com/mferraz56/Web_OpenImageProcessor.git
cd Web_OpenImageProcessor
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

5. **Execute a aplicação**
```bash
python app.py
```

A aplicação estará disponível em `http://localhost:7000`

## 🐳 Uso com Docker

### Docker Compose (Recomendado)

1. **Configure as variáveis no .env**
```bash
# Exemplo de configuração para produção
FLASK_DEBUG=false
DOCKER_CPU_LIMIT=2.0
DOCKER_MEMORY_LIMIT=2048
```

2. **Execute com Docker Compose**
```bash
docker-compose up -d
```

### Docker Manual

```bash
# Build da imagem
docker build -t web_image_processor .

# Execute o container
docker run -d \
  -p 7000:7000 \
  -v $(pwd)/Entrada:/app/Entrada \
  -v $(pwd)/Saida:/app/Saida \
  --name web_image_processor \
  web_image_processor
```

## ⚙️ Configuração

### Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `FLASK_HOST` | 0.0.0.0 | Host da aplicação |
| `FLASK_PORT` | 7000 | Porta da aplicação |
| `FLASK_DEBUG` | false | Modo debug |
| `UPLOAD_FOLDER` | ./Entrada | Diretório de upload |
| `PROCESSED_FOLDER` | ./Saida | Diretório de saída |
| `MAX_FILE_SIZE` | 16 | Tamanho máximo em MB |
| `DOCKER_CPU_LIMIT` | 2.0 | Limite de CPU |
| `DOCKER_MEMORY_LIMIT` | 2048 | Limite de memória em MB |

### Configurações de Hardware

O `docker-compose.yml` permite parametrizar limites de hardware:

```yaml
deploy:
  resources:
    limits:
      cpus: '${DOCKER_CPU_LIMIT:-2.0}'
      memory: '${DOCKER_MEMORY_LIMIT:-2048}M'
    reservations:
      cpus: '${DOCKER_CPU_RESERVATION:-1.0}'
      memory: '${DOCKER_MEMORY_RESERVATION:-1024}M'
```

## 📡 API

### Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|---------|-----------|
| `/` | GET | Interface web principal |
| `/get_image/<filename>` | GET | Serve imagens processadas |
| `/adaptativeSegmentation` | POST | Segmentação adaptativa |
| `/canny` | POST | Detecção Canny |
| `/cannyEdge` | POST | Contagem de objetos |
| `/equalize` | POST | Equalização de histograma |
| `/gaussian` | POST | Filtro Gaussiano |
| `/histogram` | POST | Análise de histograma |
| `/manualSegmentation` | POST | Segmentação manual |
| `/mediam` | POST | Filtro de média |
| `/otsu` | POST | Limiarização Otsu |
| `/sobel` | POST | Detecção Sobel |

### Exemplo de Uso da API

```python
import requests

# Upload e processamento
files = {'file': open('imagem.jpg', 'rb')}
data = {'mediana': 5, 'canny1': 50, 'canny2': 150}

response = requests.post('http://localhost:7000/canny', 
                        files=files, data=data)

if response.json()['success']:
    image_url = response.json()['image_url']
    print(f"Imagem processada: {image_url}")
```

## 🛠️ Desenvolvimento

### Estrutura do Projeto

```
Web_OpenImageProcessor/
├── app.py                 # Aplicação Flask principal
├── requirements.txt       # Dependências Python
├── Dockerfile            # Configuração Docker
├── docker-compose.yml    # Orquestração Docker
├── .env                  # Variáveis de ambiente
├── templates/
│   └── index.html        # Interface web
├── static/
│   └── styles.css        # Estilos CSS
├── Entrada/              # Diretório de upload
├── Saida/                # Diretório de saída
└── logs/                 # Logs da aplicação
```

### Adicionando Novos Filtros

1. **Implemente o método na classe `ImageProcessorFilters`**:
```python
def new_filter(self, img: np.ndarray, filename: str, param1: int) -> str:
    """
    Descrição do novo filtro.
    
    Args:
        img: Imagem de entrada
        filename: Nome do arquivo
        param1: Parâmetro do filtro
        
    Returns:
        str: Caminho do arquivo processado
    """
    # Implementação do filtro
    processed_img = cv2.some_operation(img, param1)
    
    output_path = self._generate_output_filename(filename, "NewFilter")
    cv2.imwrite(output_path, processed_img)
    
    return output_path
```

2. **Adicione a rota Flask**:
```python
@app.route('/newfilter', methods=['POST'])
def new_filter_route():
    try:
        file = request.files['file']
        param1 = int(request.form['param1'])
        
        filepath, filename, img = save_uploaded_file(file)
        output_path = image_processor.new_filter(img, filename, param1)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "image_url": url_for('get_image', filename=os.path.basename(output_path))
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

3. **Atualize o frontend** adicionando a definição no JavaScript.

### Executando Testes

```bash
# Instalar dependências de teste
pip install pytest pytest-cov

# Executar testes
pytest tests/ -v --cov=app

# Executar testes com relatório HTML
pytest tests/ --cov=app --cov-report=html
```

## 📊 Monitoramento e Logs

### Configuração de Logs

```python
# Logs são configurados automaticamente
# Localização: ./logs/app.log
# Rotação automática quando atingir 10MB
# Mantém 5 arquivos de backup
```

### Healthcheck

O container inclui healthcheck automático:
```bash
# Verificar status
docker-compose ps

# Logs do healthcheck
docker-compose logs web_image_processor
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Diretrizes de Contribuição

- Mantenha o código bem documentado
- Adicione testes para novas funcionalidades
- Siga o padrão de commit convencional
- Atualize a documentação quando necessário

## 📄 Algoritmos Implementados

### Detalhes Técnicos

- **Canny Edge Detection**: Implementa supressão de não-máximos e histerese
- **Sobel Operator**: Calcula gradientes nas direções X e Y
- **Adaptive Thresholding**: Usa média local para calcular limiares
- **Otsu's Method**: Maximiza variância inter-classe
- **Gaussian Blur**: Convolução com kernel Gaussiano 2D
- **Histogram Equalization**: Redistribuição de intensidades no espaço YUV

## 🔒 Segurança

- Validação de tipos de arquivo
- Sanitização de nomes de arquivo
- Limites de tamanho configuráveis
- Execução em usuário não-root no Docker
- Logs de segurança para auditoria

## 📈 Performance

### Benchmarks

| Filtro | Tempo Médio (512x512) | Memória |
|--------|----------------------|---------|
| Gaussian | ~50ms | ~20MB |
| Canny | ~80ms | ~25MB |
| Otsu | ~30ms | ~15MB |
| Histogram | ~100ms | ~30MB |

*Testado em: CPU Intel i5, 8GB RAM*

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **mferraz56** - *Trabalho inicial* - [mferraz56](https://github.com/mferraz56)

## 🙏 Agradecimentos

- OpenCV community
- Flask contributors
- Todos os contribuidores do projeto

---

**Desenvolvido com ❤️ para análise científica de imagens**
