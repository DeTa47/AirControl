from PyQt5 import QtCore, QtGui, QtWidgets
from configparser import ConfigParser
from ProjectR import start_program, end_program

config = ConfigParser()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(518, 465)
        MainWindow.setMaximumSize(QtCore.QSize(518, 465))
        MainWindow.setStyleSheet("background-color: rgb(4, 0, 43);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.KeyMapPage = QtWidgets.QStackedWidget(self.centralwidget)
        self.KeyMapPage.setGeometry(QtCore.QRect(100, 50, 321, 381))
        self.KeyMapPage.setObjectName("KeyMapPage")
        self.HomePage = QtWidgets.QWidget()
        self.HomePage.setObjectName("HomePage")
        self.startButton = QtWidgets.QPushButton(self.HomePage)
        self.startButton.setGeometry(QtCore.QRect(120, 240, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.startButton.setFont(font)
        self.startButton.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                        "font-family: 'Exo 2', sans-serif;\n"
                                        "color: #fff;\n"
                                        "border-radius: 5px;\n"
                                        "display: flex;\n"
                                        "flex-direction: row;\n"
                                        "justify-content: center")
        self.startButton.setObjectName("startButton")
        self.KeyMapping = QtWidgets.QPushButton(self.HomePage)
        self.KeyMapping.setGeometry(QtCore.QRect(120, 280, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.KeyMapping.setFont(font)
        self.KeyMapping.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                        "color: #fff;\n"
                                        "border-radius: 5px;\n"
                                        "font-family: 'Exo 2', sans-serif;\n"
                                        "width: 300px;\n"
                                        "display: flex;\n"
                                        "flex-direction: row;\n"
                                        "justify-content: center")
        self.KeyMapping.setObjectName("KeyMapping")

        self.label = QtWidgets.QLabel(self.HomePage)
        self.label.setGeometry(QtCore.QRect(30, 20, 261, 191))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("logo.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.quitButton = QtWidgets.QPushButton(self.HomePage)
        self.quitButton.setGeometry(QtCore.QRect(120, 320, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.quitButton.setFont(font)
        self.quitButton.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                        "font-family: 'Exo 2', sans-serif;\n"
                                        "color: #fff;\n"
                                        "border-radius: 5px;\n"
                                        "display: flex;\n"
                                        "flex-direction: row;\n"
                                        "justify-content: center")
        self.quitButton.setObjectName("quitButton")
        self.KeyMapPage.addWidget(self.HomePage)
        self.KeyMapPage_2 = QtWidgets.QWidget()
        self.KeyMapPage_2.setObjectName("KeyMapPage_2")
        self.steer_left_input = QtWidgets.QLineEdit(self.KeyMapPage_2)
        self.steer_left_input.setGeometry(QtCore.QRect(190, 150, 41, 20))
        self.steer_left_input.setStyleSheet("color: rgb(0, 0, 0);\n"
                                            "background-color: rgb(255, 255, 255);")
        self.steer_left_input.setObjectName("steer_left_input")
        self.steer_left = QtWidgets.QLabel(self.KeyMapPage_2)
        self.steer_left.setGeometry(QtCore.QRect(80, 151, 101, 16))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.steer_left.setFont(font)
        self.steer_left.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                        "font-family: 'Exo 2', sans-serif;\n"
                                        "color: #fff;\n"
                                        "border-radius: 5px;\n"
                                        "display: flex;\n"
                                        "flex-direction: row;\n"
                                        "justify-content: center;")
        self.steer_left.setObjectName("steer_left")
        self.rf_pedal = QtWidgets.QLabel(self.KeyMapPage_2)
        self.rf_pedal.setGeometry(QtCore.QRect(80, 85, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.rf_pedal.setFont(font)
        self.rf_pedal.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                    "font-family: 'Exo 2', sans-serif;\n"
                                    "color: #fff;\n"
                                    "border-radius: 5px;\n"
                                    "display: flex;\n"
                                    "flex-direction: row;\n"
                                    "justify-content: center;")
        self.rf_pedal.setObjectName("rf_pedal")
        self.steer_right = QtWidgets.QLabel(self.KeyMapPage_2)
        self.steer_right.setGeometry(QtCore.QRect(80, 180, 101, 16))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.steer_right.setFont(font)
        self.steer_right.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                        "font-family: 'Exo 2', sans-serif;\n"
                                        "color: #fff;\n"
                                        "border-radius: 5px;\n"
                                        "display: flex;\n"
                                        "flex-direction: row;\n"
                                        "justify-content: center;")
        self.steer_right.setObjectName("steer_right")
        self.lf_pedal = QtWidgets.QLabel(self.KeyMapPage_2)
        self.lf_pedal.setGeometry(QtCore.QRect(80, 120, 101, 16))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.lf_pedal.setFont(font)
        self.lf_pedal.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                    "font-family: 'Exo 2', sans-serif;\n"
                                    "color: #fff;\n"
                                    "border-radius: 5px;\n"
                                    "display: flex;\n"
                                    "flex-direction: row;\n"
                                    "justify-content: center;")
        self.lf_pedal.setObjectName("lf_pedal")
        self.steer_right_input = QtWidgets.QLineEdit(self.KeyMapPage_2)
        self.steer_right_input.setGeometry(QtCore.QRect(190, 180, 41, 20))
        self.steer_right_input.setStyleSheet("color: rgb(0, 0, 0);\n"
                                                "background-color: rgb(255, 255, 255);")
        self.steer_right_input.setObjectName("steer_right_input")
        self.lf_pedal_input = QtWidgets.QLineEdit(self.KeyMapPage_2)
        self.lf_pedal_input.setGeometry(QtCore.QRect(190, 120, 41, 20))
        self.lf_pedal_input.setStyleSheet("color: rgb(0, 0, 0);\n"
                                            "background-color: rgb(255, 255, 255);")
        self.lf_pedal_input.setObjectName("lf_pedal_input")
        self.rf_pedal_input = QtWidgets.QLineEdit(self.KeyMapPage_2)
        self.rf_pedal_input.setGeometry(QtCore.QRect(190, 90, 41, 20))
        self.rf_pedal_input.setStyleSheet("color: rgb(0, 0, 0);\n"
                                            "background-color: rgb(255, 255, 255);")
        self.rf_pedal_input.setObjectName("rf_pedal_input")
        self.back = QtWidgets.QPushButton(self.KeyMapPage_2)
        self.back.setGeometry(QtCore.QRect(20, 20, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.back.setFont(font)
        self.back.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                "font-family: 'Exo 2', sans-serif;\n"
                                "color: #fff;\n"
                                "border-radius: 5px;\n"
                                "display: flex;\n"
                                "flex-direction: row;\n"
                                "padding: 4px;")
        self.back.setObjectName("back")
        self.Save = QtWidgets.QPushButton(self.KeyMapPage_2)
        self.Save.setGeometry(QtCore.QRect(200, 240, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Exo 2")
        font.setBold(True)
        font.setWeight(75)
        self.Save.setFont(font)
        self.Save.setStyleSheet("background-color: rgb(0, 85, 255);\n"
                                "font-family: 'Exo 2', sans-serif;\n"
                                " color: #fff;\n"
                                " border-radius: 5px;\n"
                                "display: flex;\n"
                                "flex-direction: row;\n"
                                "padding: 4px;")
        self.Save.setObjectName("Save")
        self.KeyMapPage.addWidget(self.KeyMapPage_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)  

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.KeyMapping.setText(_translate("MainWindow", "Key Mapping"))
        self.quitButton.setText(_translate("MainWindow", "Quit"))
        self.steer_left.setText(_translate("MainWindow", "Steer Left"))
        self.rf_pedal.setText(_translate("MainWindow", "Right Foot Pedal"))
        self.steer_right.setText(_translate("MainWindow", "Steer Right"))
        self.lf_pedal.setText(_translate("MainWindow", "Left Foot Pedal"))
        self.back.setText(_translate("MainWindow", "Back"))
        self.Save.setText(_translate("MainWindow", "Save"))
     
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.startButton.clicked.connect(start_program)
        self.quitButton.clicked.connect(end_program)
        self.Save.clicked.connect(self.save_values)
        self.KeyMapping.clicked.connect(self.show_key_mapping_page)
        self.back.clicked.connect(self.show_home_page)

    def show_key_mapping_page(self):
        self.KeyMapPage.setCurrentIndex(1)

    def show_home_page(self):
        self.KeyMapPage.setCurrentIndex(0)
    
    def save_values(self):

        right_foot_pedal = self.rf_pedal_input.text()
        left_foot_pedal = self.lf_pedal_input.text()
        steer_left = self.steer_left_input.text()
        steer_right = self.steer_right_input.text()

        config.read("control_config.ini")

        config['CONTROLS']['rf_pedal'] = right_foot_pedal
        config['CONTROLS']['lf_pedal'] = left_foot_pedal
        config['CONTROLS']['steer_left'] = steer_left
        config['CONTROLS']['steer_right'] = steer_right

        with open("control_config.ini", "w") as f:
            config.write(f)
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())