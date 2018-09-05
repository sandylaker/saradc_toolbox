import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt
from saradc import SarAdc as Adc
from saradc_differential import SarAdcDifferential as AdcDiff
from sympy import randprime


# noinspection PyArgumentList
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.adc_args = {
            'vref': 1.2,
            'n': 12,
            'mismatch': 0.001,
            'structure': 'conventional',
        }
        self.adc_diff_args = {
            'vref': 1.2,
            'n': 12,
            'mismatch': 0.001,
            'structure': 'differential'
        }

        self.fs = 50e6
        self.fft_length = 4096
        self.prime_number = 1193
        self.vin = 0.6
        self.switch = 'conventional'
        self.resolution = 0.1
        self.method = 'fast'
        self.switch_map = {'conventional': 'conventional',
                           'monotonic': 'monotonic',
                           'mcs': 'mcs',
                           'split capacitor': 'split'}
        self.method_map = {'fast': 'fast',
                           'iterative': 'iterative',
                           'code density': 'code_density'}
        self.unit_map = {'MHz': 1e6,
                         'kHz': 1e3,
                         'Hz': 1}
        self.n_monte_carlo = 1000

        # a single-ended adc instance and
        # a adc instance with differential structure
        self.adc = Adc(**self.adc_args)
        self.adc_diff = AdcDiff(**self.adc_diff_args)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the 'figure'
        # it takes the 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to burst mode plot
        self.button_bm = QPushButton('Plot \n Burst Mode')
        self.button_bm.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_bm.clicked.connect(self.plot_bm)
        self.button_bm.setEnabled(False)

        # Just some button connected to fft plot
        self.button_fft = QPushButton('Plot \n FFT')
        self.button_fft.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_fft.clicked.connect(self.plot_fft)

        # Just some button connected to nonlinearity plot
        self.button_nonlinearity = QPushButton('Plot \n Nonlinearity')
        self.button_nonlinearity.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_nonlinearity.clicked.connect(self.plot_nonlinearity)

        # Just some button connected to energy consumption plot
        self.button_energy = QPushButton('Plot Energy \n Consumption')
        self.button_energy.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_energy.clicked.connect(self.plot_energy)
        
        # Just some button connected to continuous plot
        self.button_plot_cm = QPushButton('Plot Continuous \n Mode')
        self.button_plot_cm.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_plot_cm.clicked.connect(self.plot_cm)

        # Just button connected to input frequency selection
        self.button_fin = QPushButton('Select \n Input Frequency')
        self.button_fin.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_fin.clicked.connect(self.update_fin)

        # Just some button connected to monte carlo of dnl and inl
        self.button_mc_nl = QPushButton('Monte Carlo \n of DNL and INL')
        self.button_mc_nl.setFont(QFont('Arial', weight=QFont.Bold))
        self.button_mc_nl.clicked.connect(self.plot_monte_carlo_nl)

        # widgets to initialize the parameters of adc
        self.vref_widget = QLineEdit()
        validator_vref = QDoubleValidator()
        self.vref_widget.setValidator(validator_vref)
        self.vref_widget.setPlaceholderText(str(self.adc.vref))
        self.vref_widget.textChanged.connect(self.update_vref)

        self.label_vref = QLabel()
        self.label_vref.setText('Reference Voltage')

        self.n_widget = QSpinBox()
        self.n_widget.setMinimum(3)
        self.n_widget.setMaximum(30)
        self.n_widget.setValue(self.adc.n)
        self.n_widget.valueChanged.connect(self.update_n)

        self.label_n = QLabel()
        self.label_n.setText('Number of Bits')

        self.mismatch_widget = QLineEdit()
        self.mismatch_widget.setValidator(
            QDoubleValidator(self.mismatch_widget))
        self.mismatch_widget.setPlaceholderText(str(self.adc.mismatch))
        self.mismatch_widget.textChanged.connect(self.update_mismatch)

        self.label_mismatch = QLabel()
        self.label_mismatch.setText('Mismatch')

        self.structure_widget = QComboBox()
        self.structure_widget.addItem('conventional')
        self.structure_widget.addItem('differential')
        self.structure_widget.addItem('split capacitor array')
        self.structure_widget.currentIndexChanged.connect(self.update_structure)

        self.label_structure = QLabel()
        self.label_structure.setText('Structure')

        # widgets to initialize the parameter of burst mode
        self.label_bm = QLabel()
        self.label_bm.setText('Burst Mode:')
        self.label_bm.setFont(QFont('Arial', weight=QFont.Bold))

        self.vin_widget = QLineEdit()
        self.vin_widget.setValidator(QDoubleValidator(self.vin_widget))
        self.vin_widget.setPlaceholderText(str(0.6))
        self.vin_widget.textChanged.connect(self.update_vin)
        self.label_vin = QLabel()
        self.label_vin.setText('Input Voltage')

        self.switch_widget = QComboBox()
        self.switch_widget.addItem('conventional')
        self.switch_widget.addItem('monotonic')
        self.switch_widget.addItem('mcs')
        self.switch_widget.addItem('split capacitor')
        self.switch_widget.currentIndexChanged.connect(self.update_switch)

        self.label_switch = QLabel()
        self.label_switch.setText('Switch')

        # widgets to initialize the parameter of nonlinearity plot
        self.label_plot_nl = QLabel()
        self.label_plot_nl.setText('Nonlinearity Plot: ')
        self.label_plot_nl.setFont(QFont('Arial', weight=QFont.Bold))

        self.resolution_widget = QLineEdit()
        self.resolution_widget.setValidator(
            QDoubleValidator(self.resolution_widget))
        self.resolution_widget.setPlaceholderText(str(0.1))
        self.resolution_widget.textChanged.connect(self.update_resolution)

        self.label_resolution = QLabel()
        self.label_resolution.setText('Resolution of \n Test Signal (in LSB)')

        self.method_widget = QComboBox()
        self.method_widget.addItem('fast')
        self.method_widget.addItem('iterative')
        self.method_widget.addItem('code density')
        self.method_widget.currentIndexChanged.connect(self.update_method)

        self.label_method = QLabel()
        self.label_method.setText('Method')

        # widgets to initialize the parameter of FFT Plot
        self.label_plot_fft = QLabel()
        self.label_plot_fft.setText('FFT Plot: ')
        self.label_plot_fft.setFont(QFont('Arial', weight=QFont.Bold))

        self.fftlength_widget = QLineEdit()
        self.fftlength_widget.setValidator(QIntValidator(self.fftlength_widget))
        self.fftlength_widget.setPlaceholderText(str(self.fft_length))
        self.fftlength_widget.textChanged.connect(self.update_fft_length)

        self.label_fft_length = QLabel()
        self.label_fft_length.setText('FFT Length')

        self.fin_widget = QLineEdit()
        self.fin_widget.setPlaceholderText(str(self.prime_number/self.fft_length * self.fs))
        self.fin_widget.setReadOnly(True)

        self.label_fin = QLabel()
        self.label_fin.setText('Input Frequency')

        self.fs_widget = QLineEdit()
        self.fs_widget.setValidator(QDoubleValidator(self.fs_widget))
        self.fs_widget.setPlaceholderText(str(self.fs / 1e6))
        self.fs_widget.textChanged.connect(self.update_fs)

        # the unit of the sampling frequency
        self.fs_unit_widget = QComboBox()
        self.fs_unit_widget.addItem('MHz')
        self.fs_unit_widget.addItem('kHz')
        self.fs_unit_widget.addItem('Hz')
        self.fs_unit_widget.currentIndexChanged.connect(self.update_fs_unit)

        self.label_fs = QLabel()
        self.label_fs.setText('Sampling Frequency')

        self.label_adc_param = QLabel()
        self.label_adc_param.setText('ADC Parameters:')
        self.label_adc_param.setFont(QFont('Arial', weight=QFont.Bold))

        # set layout
        # the outermost layout
        layout = QHBoxLayout()

        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        layout.addLayout(canvas_layout)

        # layout for buttons
        grid_button = QGridLayout()
        grid_button.addWidget(self.button_bm, 1, 1)
        grid_button.addWidget(self.button_fft, 1, 2)
        grid_button.addWidget(self.button_nonlinearity, 1, 3)
        grid_button.addWidget(self.button_energy, 1, 4)
        grid_button.addWidget(self.button_fin, 2, 1)
        grid_button.addWidget(self.button_plot_cm, 2, 2)
        grid_button.addWidget(self.button_mc_nl, 2, 3)
        # insert layout of buttons into the layout of canvas
        canvas_layout.addLayout(grid_button)

        # layout for parameters
        param_layout = QGridLayout()
        param_layout.addWidget(self.label_adc_param, 1, 1, 1, 2, Qt.AlignHCenter)
        param_layout.addWidget(self.label_vref, 2, 1)
        param_layout.addWidget(self.vref_widget, 3, 1)

        param_layout.addWidget(self.label_n, 2, 2)
        param_layout.addWidget(self.n_widget, 3, 2)

        param_layout.addWidget(self.label_mismatch, 4, 1)
        param_layout.addWidget(self.mismatch_widget, 5, 1)

        param_layout.addWidget(self.label_structure, 4, 2)
        param_layout.addWidget(self.structure_widget, 5, 2)

        param_layout.addWidget(self.label_bm, 6, 1, 1, 2, Qt.AlignHCenter)
        param_layout.addWidget(self.label_vin, 7, 1)
        param_layout.addWidget(self.vin_widget, 8, 1)

        param_layout.addWidget(self.label_switch, 7, 2)
        param_layout.addWidget(self.switch_widget, 8, 2)

        param_layout.addWidget(self.label_plot_nl, 9, 1, 1, 2, Qt.AlignHCenter)
        param_layout.addWidget(self.label_resolution, 10, 1)
        param_layout.addWidget(self.resolution_widget, 11, 1)
        param_layout.addWidget(self.label_method, 10, 2)
        param_layout.addWidget(self.method_widget, 11, 2)

        param_layout.addWidget(self.label_plot_fft, 12, 1, 1, 2, Qt.AlignHCenter)
        param_layout.addWidget(self.label_fft_length, 13, 1)
        param_layout.addWidget(self.fftlength_widget, 14, 1)

        param_layout.addWidget(self.label_fin, 13, 2)
        param_layout.addWidget(self.fin_widget, 14, 2)

        param_layout.addWidget(self.label_fs, 15, 1)
        param_layout.addWidget(self.fs_widget, 16, 1)
        param_layout.addWidget(self.fs_unit_widget, 16, 2)

        layout.addLayout(param_layout)
        self.setLayout(layout)

    def enable_buttons(self, bool_value):
        if bool_value:
            self.button_bm.setEnabled(True)
            self.button_fft.setEnabled(True)
            self.button_nonlinearity.setEnabled(True)
            self.button_energy.setEnabled(True)
        else:
            self.button_bm.setEnabled(False)
            self.button_fft.setEnabled(False)
            self.button_nonlinearity.setEnabled(False)
            self.button_energy.setEnabled(False)

    def update_vref(self):
        if str(self.vref_widget.text()):
            vref = float(self.vref_widget.text())
            self.adc_args['vref'] = vref
            self.adc_diff_args['vref'] = vref
            self.update_adc()
            self.button_nonlinearity.setEnabled(True)
            self.button_energy.setEnabled(True)
            self.button_fft.setEnabled(True)
        else:
            self.enable_buttons(False)

    def update_n(self):
        if str(self.n_widget.value()):
            if int(self.n_widget.value()) % 2 == 1 and self.structure_widget.currentText() == 'split capacitor array':
                self.show_dialog_7()
            else:
                n = int(self.n_widget.value())
                self.adc_args['n'] = n
                self.adc_diff_args['n'] = n
                self.update_adc()
                self.button_nonlinearity.setEnabled(True)
                self.button_energy.setEnabled(True)
                self.button_fft.setEnabled(True)
        else:
            self.enable_buttons(False)

    def update_mismatch(self):
        if str(self.mismatch_widget.text()):
            mismatch = float(self.mismatch_widget.text())
            self.adc_args['mismatch'] = mismatch
            self.adc_diff_args['mismatch'] = mismatch
            self.update_adc()
            self.button_nonlinearity.setEnabled(True)
            self.button_energy.setEnabled(True)
            self.button_fft.setEnabled(True)
        else:
            self.enable_buttons(False)

    def update_structure(self):
        if str(self.structure_widget.currentText()):
            if str(self.structure_widget.currentText()) == 'conventional':
                structure = 'conventional'
                self.adc_args['structure'] = structure
                self.update_adc()
                if self.switch_widget.currentText() == 'monotonic' \
                        or self.switch_widget.currentText() == 'mcs':
                    self.show_dialog_1()
                    self.button_bm.setEnabled(True)
                else:
                    self.button_nonlinearity.setEnabled(True)
                    self.button_energy.setEnabled(True)
                    self.button_fft.setEnabled(True)

            elif str(self.structure_widget.currentText()) == 'split capacitor array':
                if self.switch_widget.currentText() == 'monotonic' \
                        or self.switch_widget.currentText() == 'mcs':
                    self.show_dialog_1()
                    self.structure_widget.setCurrentText('conventional')
                    self.button_bm.setEnabled(True)
                if int(self.n_widget.value()) % 2 == 1:
                    self.show_dialog_7()
                    self.button_bm.setEnabled(True)
                else:
                    self.adc_args['structure'] = 'split'
                    self.update_adc()
                    self.button_nonlinearity.setEnabled(True)
                    self.button_energy.setEnabled(True)
                    self.button_fft.setEnabled(True)

            elif str(self.structure_widget.currentText()) == 'differential':
                structure = 'differential'
                self.adc_diff_args['structure'] = structure
                self.update_adc()
                self.button_nonlinearity.setEnabled(True)
                self.button_energy.setEnabled(True)
                self.button_fft.setEnabled(True)

        else:
            self.enable_buttons(False)

    def update_vin(self):
        if str(self.vin_widget.text()):
            vin = float(self.vin_widget.text())
            if vin > self.adc_args['vref'] or vin < 0:
                self.show_dialog_2()
            else:
                self.vin = vin
                self.enable_buttons(True)

    def update_switch(self):
        if str(self.switch_widget.currentText()):
            switch = str(self.switch_widget.currentText())
            if (switch == 'mcs' or switch == 'monotonic') \
                    and (str(self.structure_widget.currentText()) == 'conventional' or
                         str(self.structure_widget.currentText()) == 'split capacitor array'):
                self.show_dialog_1()
            else:
                self.switch = switch
                self.button_bm.setEnabled(True)
        else:
            self.button_bm.setEnabled(False)

    def update_resolution(self):
        if str(self.resolution_widget.text()):
            self.resolution = float(self.resolution_widget.text())
            if str(self.method_widget.currentText()) == 'code density' and self.resolution < 0.1:
                self.show_dialog_6()
            elif str(self.method_widget.currentText()) == 'iterative' and self.resolution < 0.01:
                self.show_dialog_6()
            self.button_nonlinearity.setEnabled(True)
        else:
            if not str(self.method_widget.currentText()) == 'fast':
                self.button_nonlinearity.setEnabled(False)

    def update_method(self):
        if str(self.method_widget.currentText()):
            self.method = str(self.method_widget.currentText())
            print(self.method)
            if self.method == 'code density':
                self.show_dialog_3()
                if self.resolution_widget.text() and float(self.resolution_widget.text()) < 0.1:
                    self.show_dialog_6()
            elif self.method == 'iterative':
                self.show_dialog_4()
                if self.resolution_widget.text() and float(self.resolution_widget.text()) < 0.01:
                    self.show_dialog_6()
            self.button_nonlinearity.setEnabled(True)
        else:
            self.button_nonlinearity.setEnabled(False)

    def update_fft_length(self):
        if str(self.fftlength_widget.text()):
            fft_length = int(str(self.fftlength_widget.text()))
            self.fft_length = fft_length
            self.button_fft.setEnabled(True)
        else:
            self.button_fft.setEnabled(False)

    def update_fin(self):
        # update self.prime_number and display fin
        self.prime_number = randprime(1, int(0.5*self.fft_length))
        fin = self.prime_number / self.fft_length * self.fs
        freq_unit = 'Hz'
        fin_displayed = fin
        if fin > 1e6:
            fin_displayed = fin_displayed / 1e6  # in MHz
            freq_unit = 'MHz'
        elif 1e3 < fin <= 1e6:
            fin_displayed = fin_displayed / 1e3  # in kHz
            freq_unit = 'kHz'
        else:
            freq_unit = 'Hz'
        self.fin_widget.setText('%5.3f %s' % (fin_displayed, freq_unit))
        self.button_fft.setEnabled(True)

    def update_fs(self):
        if str(self.fs_widget.text()):
            fs = int(self.fs_widget.text())
            fs_unit = self.unit_map[str(self.fs_unit_widget.currentText())]
            fs = fs * fs_unit
            print('fs: ', fs)
            self.fs = fs
            self.button_fft.setEnabled(True)
        else:
            self.button_fft.setEnabled(False)

    def update_fs_unit(self):
        if str(self.fs_unit_widget.currentText()):
            self.update_fs()

    def plot_bm(self):
        # refresh the figure
        self.figure.clear()
        switch = self.switch_map.get(self.switch)
        if str(self.structure_widget.currentText()) == 'differential':
            self.adc_diff.plot_burst_mode(self.vin, switch=switch)
        else:
            self.adc.plot_burst_mode(self.vin, switch=switch)

        self.toolbar.update()
        # refresh canvas
        self.canvas.draw()

    def plot_fft(self):
        # refresh the figure
        self.figure.clear()
        if self.prime_number > 0.5 * self.fft_length:
            self.show_dialog_5()
        else:
            if str(self.structure_widget.currentText()) == 'differential':
                self.adc_diff.plot_fft(self.fs, self.fft_length, self.prime_number)
            else:
                self.adc.plot_fft(self.fs, self.fft_length, self.prime_number)

            self.toolbar.update()
            # refresh canvas
            self.canvas.draw()

    def plot_nonlinearity(self):
        # refresh the figure
        self.figure.clear()
        method = self.method_map.get(self.method)
        if str(self.structure_widget.currentText()) == 'differential':
            self.adc_diff.plot_dnl_inl(self.resolution, method)
        else:
            self.adc.plot_dnl_inl(self.resolution, method)

        self.toolbar.update()
        # refresh canvas
        self.canvas.draw()

    def plot_energy(self):
        # refresh the figure
        self.figure.clear()
        if str(self.structure_widget.currentText()) == 'differential':
            self.adc_diff.plot_energy()
        else:
            self.adc.plot_energy()

        self.toolbar.update()
        # refresh canvas
        self.canvas.draw()
    
    def plot_cm(self):
        # refresh the figure
        self.figure.clear()
        if str(self.structure_widget.currentText()) == 'differential':
            self.adc_diff.plot_continuous_mode(self.fft_length, self.fs, self.prime_number)
        else:
            self.adc.plot_continuous_mode(self.fft_length, self.fs, self.prime_number)

        self.toolbar.update()
        # refresh canvas
        self.canvas.draw()

    def plot_monte_carlo_nl(self):
        # refresh the figure
        self.figure.clear()

        dnl_array = np.array([])
        inl_array = np.array([])
        if str(self.structure_widget.currentText()) == 'differential':
            self.update_adc()
            dnl_array = self.adc_diff.dnl(method='fast')
            inl_array = self.adc_diff.inl(dnl_array)[np.newaxis, :]
            dnl_array = dnl_array[np.newaxis, :]
            for i in range(1, self.n_monte_carlo):
                self.update_adc()
                dnl = self.adc_diff.dnl(method='fast')
                inl = self.adc_diff.inl(dnl)[np.newaxis, :]
                dnl = dnl[np.newaxis, :]
                dnl_array = np.concatenate((dnl_array, dnl))
                inl_array = np.concatenate((inl_array, inl))
        else:
            self.update_adc()
            dnl_array = self.adc.dnl(method='fast')
            inl_array = self.adc.inl(dnl_array)[np.newaxis, :]
            dnl_array = dnl_array[np.newaxis, :]
            for i in range(1,self.n_monte_carlo):
                self.update_adc()
                dnl = self.adc.dnl(method='fast')
                inl = self.adc.inl(dnl)[np.newaxis, :]
                dnl = dnl[np.newaxis, :]
                dnl_array = np.concatenate((dnl_array, dnl))
                inl_array = np.concatenate((inl_array, inl))
        print('len(dnl_array)', len(dnl_array))
        print('len(inl_array', len(inl_array))
        # plot monte carlo of dnl
        ax1 = plt.subplot(221)
        for i in range(len(dnl_array)):
            ax1.plot(np.arange(len(dnl_array[i])), dnl_array[i], linewidth=0.5)
        # ax1.set_xlabel('Output Code')
        ax1.set_ylabel('DNL (LSB)')
        ax1.set_title('DNL')
        plt.grid(True)

        sigma_dnl = np.std(dnl_array, axis=0, ddof=1)

        # plot stardard deviation of DNL
        ax2 = plt.subplot(222)
        ax2.plot(np.arange(len(sigma_dnl)), sigma_dnl, linewidth=0.5)
        # ax2.set_xlabel('Output Code')
        ax2.set_ylabel(r'$\sigma_{DNL}$ (LSB)')
        ax2.set_title(r'$\sigma$ of DNL')
        plt.grid(True)

        # plot monte carlo of inl
        ax3 = plt.subplot(223)
        for i in range(len(inl_array)):
            ax3.plot(np.arange(len(inl_array[i])), inl_array[i], linewidth=0.5)
        ax3.set_xlabel('Output Code')
        ax3.set_ylabel('INL (LSB)')
        ax3.set_title('INL')
        plt.grid(True)

        sigma_inl = np.std(inl_array, axis=0, ddof=1)
        # plot stardart deviation of INLf
        ax4 = plt.subplot(224)
        ax4.plot(np.arange(len(sigma_inl)), sigma_inl, linewidth=0.5)
        ax4.set_xlabel('Output Code')
        ax4.set_ylabel(r'$\sigma_{INL}$ (LSB)')
        ax4.set_title(r'$\sigma$ of INL')
        plt.grid(True)

        plt.tight_layout()

        self.toolbar.update()
        # refresh canvas
        self.canvas.draw()

    def update_adc(self):
        self.adc.__init__(**self.adc_args)
        self.adc_diff.__init__(**self.adc_diff_args)

    def show_dialog_1(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Conventional and split capacitor structure does not support '
                    'monotonic or mcs switching method in this toolbox. ')
        msg.setInformativeText('Please choose another switching method.')
        msg.setWindowTitle('Conflict between Structure and Switching Method')
        msg.setStandardButtons(QMessageBox.Ok)
        self.switch_widget.setCurrentText('conventional')
        ret = msg.exec_()

    def show_dialog_2(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Invalid Input Voltage.')
        msg.setInformativeText('Please enter a proper input voltage.')
        msg.setWindowTitle('Invalid Input Voltage')
        msg.setStandardButtons(QMessageBox.Ok)
        ret = msg.exec_()

    def show_dialog_3(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Runtime Warning of Code Density Method')
        msg.setInformativeText('the code density method is extremely slow,'
                               ' for 0.1 resolution and 12 bit, it might take 15 minutes.')
        msg.setWindowTitle('Runtime Warning')
        msg.setStandardButtons(QMessageBox.Ok)
        ret = msg.exec_()

    def show_dialog_4(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Runtime Warning of Iterative Method')
        msg.setInformativeText('the iterative method is very slow,'
                               ' for 0.01 resolution and 12 bit, it might take 50 seconds.')
        msg.setWindowTitle('Runtime Warning')
        msg.setStandardButtons(QMessageBox.Ok)
        ret = msg.exec_()

    def show_dialog_5(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Choose A Smaller Prime Number To Avoid Aliasing.')
        msg.setWindowTitle('Invalid prime number')
        msg.setStandardButtons(QMessageBox.Ok)
        ret = msg.exec_()

    def show_dialog_6(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Stack Overflow Warning')
        msg.setInformativeText('Too small resolution parameter might result in stack overflow, '
                               'when iterative or code density method is chosen. Please increase the'
                               ' resolution parameter, or choose the fast method,'
                               ' which is fast and precise and the resolution parameter'
                               ' is not required.')
        msg.setWindowTitle('Stack Overflow Warning')
        msg.setStandardButtons(QMessageBox.Ok)
        self.method_widget.setCurrentText('fast')
        self.resolution_widget.clear()
        self.resolution_widget.setPlaceholderText(str(0.1))
        ret = msg.exec_()

    def show_dialog_7(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText('Choose Even Number for Split Capacitor Array Structure')
        msg.setInformativeText('In the split capacitor array structure, the capacitor array is split in to '
                               'two sub-arrays,with the same number of bits. Thus, a even number of bits is '
                               'required.')
        msg.setStandardButtons(QMessageBox.Ok)
        self.n_widget.setValue(12)
        ret = msg.exec_()



if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())