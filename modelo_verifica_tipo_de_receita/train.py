import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


textos = [
    "À Nutricionista,  Encaminho o paciente supracitado, portador de HAS, DLP e obesidade 1 para avaliação e seguimento.  Grata, ",
    "Ao Funcional Paciente apresentando redução de mobilidade de quadril, tendinite de tornozelo e redução fraqueza do tibial. Solicito exercícios que foquem em mobilidade de quadril (alongamento), fortalecimento do tibial. ",
    "10 SESSÕES DE FISIOTERAPIA PARA MEMBROS INFERIORES  ENTORSE D ETORNOZELO - REABILITAÇÃO PÓS ENTORSE CÓDIGO: 20103492 CID 10: S932 JUSTIFICATIVA: ENTORSE COM LESÃO LIGAMENTAR ",
    "Ambulatório de Ortopedia - CMD Solicitação de Exames: Radiografia - Rx do Pé D AP + Obliq. OBS: Realizar Rx antes da consulta. Retorno: 26/01/2024",
    "AO LABORATÓRIO SOLICITO TESTE INFLUENZA A E B PARA PACIENTE ACIMA COM SÍNDROME GRIPAL CID J11",
    "SOLICITO HEMOGRAMA GLICEMIA JEJUM HEMOGLOBINA GLICADA TSH / T4 LIVRE COLESTEROL TOTAL E FRAÇÕES TRIGLICERÍDEOS TGO / TGP CREATININA / UREIA 25-OH-VITAMINA D HIV 1/2 / VDRL HBSAG / ANTI-HCV CID Z01.4 ",
    "ORIENTAÇÕES 1- USAR A MEDICAÇÃO CONFORME A RECEITA 2- PODE SANGRAR POR 10 DIAS 3- EVITAR RELAÇÃO NESSE PERÍODO 4- A BIÓPSIA FICA PRONTA EM 2-3 SEMANAS, EU AVISO PARA AGENDAR O RETORNO 5- DÚVIDAS, ME LIGUE ",
    "USO ORAL MAXSULID 400MG. TOMAR 01 COMPRIMIDO A CADA 12 HORAS POR 05 DIAS LISADOR DIP 1G. TOMAR 01 COMPRIMIDO A CADA 06 HORAS SE SENTIR DOR NOVOTRAM 50MG. TOMAR 01 COMPRIMIDO A CADA 06 HORAS SE SENTIR DOR INTENSA "
]

labels = [0, 0, 0, 1, 1, 1, 2, 2]  # 0 = Encaminhamento, 1 = Exame, 2 = Prescrição

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos).toarray()
y = torch.tensor(labels)

joblib.dump(vectorizer, "vectorizer_tipo_receita.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ReceitaClassifier(nn.Module):
    """
    Modelo de rede neural para classificação de receitas médicas.

    A rede contém duas camadas totalmente conectadas (fully connected)
    com uma ativação ReLU intermediária.

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


input_size = X_train.shape[1]
num_classes = 3
model = ReceitaClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Época [{epoch+1}/{num_epochs}], Perda: {loss.item():.4f}")

print("Treinamento concluído!")


torch.save(model.state_dict(), "modelo_tipo_receitas.pth")
print("Modelo salvo como 'modelo_tipo_receitas.pth'")
