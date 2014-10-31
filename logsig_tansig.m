clc;
clear;
num_entradas = 3;
num_neuronas_capa_oculta = 2;

datos_entrenamiento = csvread('entrenamiento.txt'); 
datos_entrenamiento = datos_entrenamiento';
salida = datos_entrenamiento(1, :);
entradas = datos_entrenamiento(2: num_entradas + 1, :); 

plotpv(entradas, salida);

min_max=minmax(entradas); 
net=newff(min_max,[num_neuronas_capa_oculta 1],{'logsig','tansig'},'trainlm');
net.trainParam.epochs = 100;
net.iw{1, 1}=[-1.5382   -0.7699   -3.9367; 6.4072   -0.2212   -1.0990]
net.b{1}=[2.1811; 15.9629]
net = train(net,entradas,salida); 


pesos = net.iw{1, 1};
bias =net.b{1};

plotpc(pesos, bias);

datos_test = csvread('testing.txt'); 
datos_test = datos_test';
salida_test = datos_test(1,:);
entradas_test = datos_test(2: num_entradas + 1, :); 

result = sim(net, entradas_test);
N = size(salida_test);
num_aciertos = 1;
for i=1:N(2),
    if result(i) > 0.2,
        if salida_test(i) == 1,
               num_aciertos = num_aciertos + 1;
        end
    else
       if salida_test(i) == 0,
               num_aciertos = num_aciertos + 1;
        end
    end
end  

fprintf('de 50 tipos de tumores evaluados %s la red a acertado en su conocimiento \n',num2str(num_aciertos))
