import re


palavras_chave_exames_imagem = [
    "radiografia", "raio-x", "raio x", "rx", "tomografia", "tc", "ressonância magnética", "rm", "ultrassonografia", "us", "ressonancia",
    "mamografia", "mmg", "densitometria óssea", "dmo", "angiografia", "ag", "arteriografia", "ar", "venografia", "vg",
    "fluoroscopia", "fs", "cintigrafia", "cg", "colangiopancreatografia endoscópica retrógrada", "cpre", "urografia", "ug",
    "laparoscopia", "lps", "artroscopia", "ats", "endoscopia", "es", "colonoscopia", "cc", "cistoscopia", "cs",
    "broncoscopia", "bs", "dacriocistografia", "dcs", "histerossalpingografia", "hsg", "esofagografia", "egf", 
    "videofluoroscopia de deglutição", "vfd", "raio-x", "raio x", "raios-x", "raios x", "tomografia computadorizada"
    "ultrassonografia", "ultrassom", "mamografia", "densitometria óssea", "cintigrafia", "colangiopancreatografia", "cpre", "Doppler",
    "endoscopia", "laparoscopia", "artroscopia", "broncoscopia", "histerossalpingografia", "dacriocistografia", 
    "histerossalpingografia", "colonoscopia", "cistoscopia", "videofluoroscopia", "ressonância magnética", "ressonancia magnetica", "ressonância nuclear magnética", "rnm", "ressonância magnética nuclear", "rmn", "Ecodopplercardiograma Transtorácico"
]

siglas_para_nome = {
    "us": "ultrassonografia",
    "rm": "ressonância magnética",
    "tc": "tomografia computadorizada",
    "dmo": "densitometria óssea",
    "mmg": "mamografia",
    "ag": "angiografia",
    "ar": "arteriografia",
    "vg": "venografia",
    "fs": "fluoroscopia",
    "cg": "cintigrafia",
    "cpre": "colangiopancreatografia endoscópica retrógrada",
    "ug": "urografia",
    "lps": "laparoscopia",
    "ats": "artroscopia",
    "es": "endoscopia",
    "cc": "colonoscopia",
    "cs": "cistoscopia",
    "bs": "broncoscopia",
    "dcs": "dacriocistografia",
    "hsg": "histerossalpingografia",
    "egf": "esofagografia",
    "vfd": "videofluoroscopia de deglutição",
    "rnm": "ressonância nuclear magnética",
    "rmn": "ressonância magnética nuclear"
}

def retorna_se_e_exame_de_imagem_e_termos_encontrados(text):
    """
    Verifica se o texto contém termos relacionados a exames de imagem e retorna os exames encontrados.
    
    Parâmetros:
        text (str): Texto a ser analisado.
    
    Retorna:
        list: Lista de exames de imagem identificados no texto.
    """
    text = text.lower()
    exames_encontrados = []
    
    for term in palavras_chave_exames_imagem:
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            if match in siglas_para_nome:
                exames_encontrados.append(siglas_para_nome[match])
            else:
                exames_encontrados.append(match)
    
    return exames_encontrados


# texto_exemplo = "Solicito RNM do tornozelo D para a paciente acima. HD = dor no tornozelo D a esclarecer CID M25 Paciente teve fratura exposta da tíbia distal D Salter II há 6 anos, operada mas aparentemente apresentou boa evolução radiográfica. Grata"
# exames = retorna_se_e_exame_de_imagem_e_termos_encontrados(texto_exemplo)

# if exames:
#     print(f"Foi pedido exame de imagem? Sim - Exames encontrados: {', '.join(exames)}")
# else:
#     print("Nenhum exame de imagem encontrado na Receita")
