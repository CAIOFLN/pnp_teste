#!/usr/bin/env python3

import rospy
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from tf.transformations import quaternion_from_euler

class ImagePublisherAndInferenceNode:
    def __init__(self):
        # Inicializa o nó ROS
        rospy.init_node('image_publisher_inference_node', anonymous=True)
        
        # Parâmetros
        self.model_path = rospy.get_param('~model_path', 'model.best')  # Caminho do modelo .best
        self.output_topic = rospy.get_param('~output_topic', '/image_inference/output')  # Tópico da imagem processada

        # Carregar modelo YOLO
        self.model = YOLO(self.model_path)

        # Configura o CvBridge para conversão entre OpenCV e ROS
        self.bridge = CvBridge()

        # Configura o publicador da imagem processada
        self.pub_image = rospy.Publisher(self.output_topic, Image, queue_size=10)

        # Configura a captura da webcam
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            rospy.logerr("Erro ao abrir a câmera.")
            return
        
        # Configurar a resolução da câmera para 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Parâmetros da câmera (substitua pelos valores de calibração da sua câmera)
        self.dist_coeffs = np.array([0.075121, -0.135470, 0.000294, -0.003627, 0.000000])  # Coeficientes de distorção
        self.camera_matrix = np.array([[611.771006, 0.000000, 259.623778],
                                       [0.000000, 610.311411, 255.611100],
                                       [0.000000, 0.000000, 1.000000]], dtype=np.float32)

        rospy.loginfo("Image Publisher and Inference Node iniciado com sucesso.")

    def run(self):
        rate = rospy.Rate(5)  # Publica a imagem a 1 Hz para reduzir a carga
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Erro ao capturar a imagem.")
                break

            # Executar a inferência
            results = self.model.predict(frame)

            # Variáveis para armazenar coordenadas dos keypoints
            keypoints = {}

            # Obter os keypoints, se houver
            if results[0].keypoints is not None:
                # Extraímos as coordenadas dos keypoints
                for i, kp in enumerate(results[0].keypoints.xy[0], start=1):  # Para cada keypoint, extraímos (x, y)
                    x, y = int(kp[0]), int(kp[1])
                    keypoints[f'KP{i}'] = (x, y)
                    print(f"Keypoint {i} detectado: ({x}, {y})")
                    
                    # Desenhar o ponto
                    cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.putText(frame, f'KP{i}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Verificar se temos os pontos KP2, KP3, KP4 e KP5
                if all(k in keypoints for k in ['KP2', 'KP3', 'KP4', 'KP5']):
                    print("Todos os keypoints necessários foram detectados.")
                    
                    # Definir coordenadas 3D dos pontos em relação ao centro da cruz
                    crossArmLength = 0.245  # Substitua com o valor real
                    object_points = np.array([
                        [0, crossArmLength, 0],    # KP2 (topo da cruz)
                        [crossArmLength, 0, 0],    # KP3 (direita da cruz)
                        [0, -crossArmLength, 0],   # KP4 (base da cruz)
                        [-crossArmLength, 0, 0]    # KP5 (esquerda da cruz)
                    ], dtype=np.float32)

                    # Coordenadas dos keypoints na imagem
                    image_points = np.array([
                        keypoints['KP2'],
                        keypoints['KP3'],
                        keypoints['KP4'],
                        keypoints['KP5']
                    ], dtype=np.float32)

                    # Calcular a pose usando solvePnP
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:
                        # Exibir a posição no terminal
                        print(f"Posição do centro da cruz em relação à câmera: x={tvec[0][0]:.2f}, y={tvec[1][0]:.2f}, z={tvec[2][0]:.2f}")

            # Converter a imagem processada para o formato de mensagem ROS
            try:
                img_msg_out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                # Publicar a imagem processada
                self.pub_image.publish(img_msg_out)
            except CvBridgeError as e:
                rospy.logerr(f"Erro na conversão da imagem: {e}")

            rate.sleep()

        self.cap.release()

if __name__ == '__main__':
    try:
        node = ImagePublisherAndInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
