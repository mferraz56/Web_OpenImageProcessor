#!/usr/bin/env python3
"""
Script de teste para validar as funcionalidades do processador de imagens.

Este script pode ser usado para testar as funcionalidades básicas sem a interface web.
"""

import os
import sys
import cv2
import numpy as np
from app import ImageProcessorFilters

def create_test_image():
    """Cria uma imagem de teste simples."""
    # Criar uma imagem 400x400 com alguns elementos geométricos
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Fundo gradiente
    for i in range(400):
        img[i, :, 0] = int(255 * i / 400)  # Canal azul
    
    # Círculo branco
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    
    # Retângulo cinza
    cv2.rectangle(img, (200, 200), (300, 300), (128, 128, 128), -1)
    
    # Triângulo
    triangle = np.array([[150, 350], [250, 300], [350, 350]], np.int32)
    cv2.fillPoly(img, [triangle], (200, 100, 50))
    
    return img

def test_filters():
    """Testa todos os filtros disponíveis."""
    print("🧪 Iniciando testes dos filtros de processamento de imagem...")
    
    # Criar instância do processador
    processor = ImageProcessorFilters()
    
    # Criar imagem de teste
    test_img = create_test_image()
    test_filename = "test_image.jpg"
    
    # Salvar imagem de teste
    test_path = os.path.join("Entrada", test_filename)
    cv2.imwrite(test_path, test_img)
    print(f"✅ Imagem de teste criada: {test_path}")
    
    # Lista de testes
    tests = [
        ("Segmentação Adaptativa", lambda: processor.adaptive_threshold_segmentation(test_img, test_filename, 5)),
        ("Canny Edge Detection", lambda: processor.canny_edge_detection(test_img, test_filename, 5, 50, 150)),
        ("Contagem de Objetos", lambda: processor.object_counting_with_contours(test_img, test_filename)),
        ("Equalização de Histograma", lambda: processor.histogram_equalization(test_img, test_filename)),
        ("Filtro Gaussiano", lambda: processor.gaussian_blur(test_img, test_filename, 2, 7)),
        ("Análise de Histograma", lambda: processor.generate_histogram_plot(test_img, test_filename)),
        ("Segmentação Manual", lambda: processor.manual_threshold_segmentation(test_img, test_filename, 5, 100, 255)),
        ("Filtro de Média", lambda: processor.mean_blur_filter(test_img, test_filename, 5)),
        ("Limiarização Otsu", lambda: processor.otsu_threshold(test_img, test_filename)),
        ("Sobel Edge Detection", lambda: processor.sobel_edge_detection(test_img, test_filename, 1, 1, 3)),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"🔄 Testando: {test_name}")
            output_path = test_func()
            
            if os.path.exists(output_path):
                print(f"✅ {test_name}: SUCESSO - {output_path}")
                results.append((test_name, "SUCESSO", output_path))
            else:
                print(f"❌ {test_name}: FALHA - Arquivo não criado")
                results.append((test_name, "FALHA", "Arquivo não encontrado"))
                
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {str(e)}")
            results.append((test_name, "ERRO", str(e)))
    
    # Resumo dos resultados
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    sucessos = 0
    for test_name, status, details in results:
        status_icon = "✅" if status == "SUCESSO" else "❌"
        print(f"{status_icon} {test_name:25} | {status}")
        if status == "SUCESSO":
            sucessos += 1
    
    print(f"\n📈 Resultado Final: {sucessos}/{len(tests)} testes passaram")
    
    if sucessos == len(tests):
        print("🎉 Todos os testes passaram com sucesso!")
        return True
    else:
        print("⚠️  Alguns testes falharam. Verifique os logs acima.")
        return False

def validate_environment():
    """Valida se o ambiente está configurado corretamente."""
    print("🔍 Validando ambiente...")
    
    issues = []
    
    # Verificar diretórios
    dirs_to_check = ["Entrada", "Saida", "logs"]
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            issues.append(f"Diretório não encontrado: {directory}")
        else:
            print(f"✅ Diretório encontrado: {directory}")
    
    # Verificar dependências
    try:
        import cv2
        print(f"✅ OpenCV versão: {cv2.__version__}")
    except ImportError:
        issues.append("OpenCV não instalado")
    
    try:
        import flask
        print(f"✅ Flask versão: {flask.__version__}")
    except ImportError:
        issues.append("Flask não instalado")
    
    try:
        import numpy
        print(f"✅ NumPy versão: {numpy.__version__}")
    except ImportError:
        issues.append("NumPy não instalado")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib versão: {matplotlib.__version__}")
    except ImportError:
        issues.append("Matplotlib não instalado")
    
    if issues:
        print("\n❌ Problemas encontrados:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("\n✅ Ambiente validado com sucesso!")
        return True

def main():
    """Função principal do script de teste."""
    print("🖼️  Web OpenImage Processor - Script de Teste")
    print("=" * 50)
    
    # Validar ambiente
    if not validate_environment():
        print("\n❌ Falha na validação do ambiente. Corrija os problemas antes de continuar.")
        sys.exit(1)
    
    # Executar testes
    print("\n" + "=" * 50)
    if test_filters():
        print("\n🎉 Todos os testes concluídos com sucesso!")
        print("💡 Você pode agora executar 'python app.py' para iniciar a aplicação web.")
        sys.exit(0)
    else:
        print("\n⚠️  Alguns testes falharam. Verifique a implementação.")
        sys.exit(1)

if __name__ == "__main__":
    main()
