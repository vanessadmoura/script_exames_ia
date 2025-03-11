import torch
import torch.nn as nn
import joblib


vectorizer = joblib.load("vectorizer_tipo_receita.pkl")

class ReceitaClassifier(nn.Module):
    """
    Modelo de rede neural para classificação de tipos de receitas médicas.

    A rede contém duas camadas totalmente conectadas com ativação ReLU intermediária.

    Parâmetros:
        input_size (int): Número de características de entrada.
        num_classes (int): Número de classes de saída.

    Métodos:
        forward(x): Propaga os dados pela rede e retorna as previsões.
    """

    def __init__(self, input_size, num_classes):
        super(ReceitaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
    

    def forward(self, x):
        """
        Propaga os dados pela rede e retorna as previsões.

        Parâmetros:
            x (Tensor): Tensor de entrada com características vetorizadas.

        Retorna:
            Tensor: Saída com as previsões do modelo.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


input_size = len(vectorizer.get_feature_names_out())  
num_classes = 3
model = ReceitaClassifier(input_size, num_classes)
model.load_state_dict(torch.load("modelo_tipo_receitas.pth"))
model.eval()  


def identificar_tipo_de_receita(receita):
    """
    Classifica um texto de receita médica em uma das categorias: 
    'Encaminhamento', 'Exame Médico' ou 'Prescrição'.

    Parâmetros:
        receita (str): Texto contendo a receita médica.

    Retorna:
        str: Categoria classificada do texto.
    """
    if not isinstance(receita, str):
        raise ValueError("A receita fornecida não é uma string válida.")
    
    X_novo = vectorizer.transform([receita]).toarray()
    X_novo_tensor = torch.tensor(X_novo, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_novo_tensor)
    
    pred = torch.argmax(output).item()
    
    categorias = {0: "Encaminhamento", 1: "Exame Médico", 2: "Prescrição"}
    return categorias[pred]


# print(identificar_tipo_de_receita("A nutricionista Encaminho paciente para avaliação e conduta. Grata"))
# print(identificar_tipo_de_receita("Solicito hemograma U Cr TGO TGP FALc GGT hb glic"))
# print(identificar_tipo_de_receita("USO ORAL: 1) NAPROXENO 550 MG ---------------- TOMAR UM COMPRIMIDO DE 12 EM 12 HORAS SE DOR 2) ENTEROGERMINA PLUS ----------------- TOMAR UM FLACONETE AO DIA POR 5 DIAS  "))