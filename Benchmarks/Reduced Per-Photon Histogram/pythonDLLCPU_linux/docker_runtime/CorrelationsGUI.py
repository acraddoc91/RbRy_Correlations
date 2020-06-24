# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CorrelationsGUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5.uic import loadUiType
from PyQt5 import QtWidgets
from PyQt5.QtGui import QDoubleValidator, QTextCursor
from PyQt5 import QtCore
from time import sleep
from subprocess import call
from os import getcwd

Ui_MainWindow,QMainWindow = loadUiType('CorrelationsGUI.ui')

class textStream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)
    def write(self, text):
        self.newText.emit(str(text))
    def flush(self):
        pass

def processFiles(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False):
    call_string = "command " + str(g2_proccessing) + " " + str(g3_proccessing) + " \"" + str(folder_name) + "\" \"" + str(file_out_name) + "\" " + str(max_time) + " " + str(bin_width) + " " + str(pulse_spacing) + " " + str(max_pulse_distance) + " " + str(calc_norm) + " " + str(update)
    mount_dir = getcwd() + "\\auth"
    docker_string = "docker run --rm --privileged --mount type=bind,source=\"" + mount_dir + "\",target=/home/rbry/auth acraddoc91/correlations "
    call(docker_string + call_string, shell=True)

class processingThread(QtCore.QThread):
    do_processing = False
    do_g2 = False
    do_g3 = False
    do_continuous = False
    folder_name = ""
    file_out_name = ""
    max_time = 1e-6
    bin_width = 1e-9
    pulse_spacing = 100e-6
    max_pulse_distance = 4
    continuous = False
    update = False

    def __init__(self,g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update,continuous):
        QtCore.QThread.__init__(self)
        self.do_g2 = g2_proccessing
        self.do_g3 = g3_proccessing
        self.folder_name = folder_name
        self.file_out_name = file_out_name
        self.max_time = max_time
        self.bin_width = bin_width
        self.pulse_spacing = pulse_spacing
        self.max_pulse_distance = max_pulse_distance
        self.update = update
        self.do_continuous = continuous
        self.do_processing = True
    def __del__(self):
        pass
    def run(self):
        while self.do_processing:
            processFiles(self.do_g2,self.do_g3,self.folder_name,self.file_out_name,self.max_time,self.bin_width,self.pulse_spacing,self.max_pulse_distance,True,self.update)
            if not self.do_continuous:
                self.do_processing = False
            else:
                sleep(30)
    def terminate(self):
        self.do_processing = False

class Main(QMainWindow,Ui_MainWindow):
    working = False
    def __init__(self):
        super(Main,self).__init__()
        self.setupUi(self)
        self.getDataFolder.clicked.connect(self.findDataFolderFunc)
        self.getOutputFilename.clicked.connect(self.findOutputFileFunc)
        self.valid_bin_width = QDoubleValidator()
        self.valid_tau = QDoubleValidator()
        self.binWidth.setValidator(self.valid_bin_width)
        self.maxTau.setValidator(self.valid_tau)
        sys.stdout = textStream(newText = self.updateStream)
        self.startButton.clicked.connect(self.startProcessing)
        self.stopButton.clicked.connect(self.stopProcessing)


    def findDataFolderFunc(self):
        self.dataFolderName.setText(QtWidgets.QFileDialog.getExistingDirectory())

    def findOutputFileFunc(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(None,"Title","","*.mat")
        self.outputFilename.setText(filename)

    def updateStream(self,text):
        cursor = self.outputText.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)

    def startProcessing(self):
        if not self.working:
            do_g2 = self.g2Button.isChecked()
            do_g3 = self.g3Button.isChecked()
            do_update = self.updateButton.isChecked()
            folder_name = self.dataFolderName.text()
            if folder_name[-1] != '/':
                folder_name = folder_name + '/'
            print("Processing")
            self.working = True
            self.thread = processingThread(do_g2, do_g3, folder_name, self.outputFilename.text() ,float(self.maxTau.text()), float(self.binWidth.text()), 100e-6, 4, do_update, self.continuousButton.isChecked())
            self.thread.finished.connect(self.doneProcessing)
            self.thread.start()
    def doneProcessing(self):
        print("All done")
        self.working = False
    def stopProcessing(self):
        if self.working:
            self.thread.terminate()
if __name__=='__main__':
    import sys
    from PyQt5 import QtGui
    
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())