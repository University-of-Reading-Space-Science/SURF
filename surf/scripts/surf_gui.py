"""PyQt GUI for configuring and running SURF workflows."""

import datetime
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import astropy.units as u
import matplotlib.pyplot as plt
from PyQt6.QtCore import QDateTime, QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from sunpy.coordinates import sun

import surf.surf_analysis as sa
import surf.surf_inputs as sin


class ModelParametersTab(QWidget):
    """Tab containing core model domain and coordinate settings."""

    start_datetime_updated = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        model_box = QGroupBox("Model Parameters")
        form = QFormLayout()

        self.rmin_spin = QDoubleSpinBox()
        self.rmin_spin.setRange(0.0, 1000.0)
        self.rmin_spin.setSingleStep(0.1)
        self.rmin_spin.setValue(21.5)
        self.rmin_spin.setSuffix(" Rs")

        self.rmax_spin = QDoubleSpinBox()
        self.rmax_spin.setRange(0.0, 1000.0)
        self.rmax_spin.setSingleStep(0.1)
        self.rmax_spin.setValue(240.0)
        self.rmax_spin.setSuffix(" Rs")

        self.lon_min_spin = QDoubleSpinBox()
        self.lon_min_spin.setRange(-360.0, 360.0)
        self.lon_min_spin.setSingleStep(1.0)
        self.lon_min_spin.setValue(0.0)
        self.lon_min_spin.setSuffix(" deg")

        self.lon_max_spin = QDoubleSpinBox()
        self.lon_max_spin.setRange(-360.0, 360.0)
        self.lon_max_spin.setSingleStep(1.0)
        self.lon_max_spin.setValue(360.0)
        self.lon_max_spin.setSuffix(" deg")

        self.latitude_spin = QDoubleSpinBox()
        self.latitude_spin.setRange(-90.0, 90.0)
        self.latitude_spin.setSingleStep(1.0)
        self.latitude_spin.setValue(0.0)
        self.latitude_spin.setSuffix(" deg")

        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["sidereal", "synodic"])
        self.frame_combo.setCurrentText("sidereal")

        self.one_d_toggle = QCheckBox("Run as 1D (single longitude)")
        self.one_d_toggle.toggled.connect(self._on_1d_toggled)

        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_datetime.setTimeSpec(Qt.TimeSpec.UTC)
        self.start_datetime.setDateTime(QDateTime.currentDateTimeUtc())

        self.cr_num_spin = QSpinBox()
        self.cr_num_spin.setRange(1, 10000)

        self.cr_lon_init_spin = QDoubleSpinBox()
        self.cr_lon_init_spin.setRange(0.0, 360.0)
        self.cr_lon_init_spin.setSingleStep(1.0)
        self.cr_lon_init_spin.setSuffix(" deg")

        self.start_equivalent_label = QLabel("")
        self.start_equivalent_label.setWordWrap(True)

        self.start_datetime.dateTimeChanged.connect(self._sync_from_datetime)
        self.cr_num_spin.valueChanged.connect(self._sync_from_carrington)
        self.cr_lon_init_spin.valueChanged.connect(self._sync_from_carrington)

        lon_row = QWidget()
        lon_layout = QHBoxLayout()
        lon_layout.setContentsMargins(0, 0, 0, 0)
        lon_layout.addWidget(self.lon_min_spin)
        lon_layout.addWidget(QLabel("to"))
        lon_layout.addWidget(self.lon_max_spin)
        lon_row.setLayout(lon_layout)

        form.addRow("rmin", self.rmin_spin)
        form.addRow("rmax", self.rmax_spin)
        form.addRow("Run mode", self.one_d_toggle)
        form.addRow("Longitude range", lon_row)
        form.addRow("Latitude", self.latitude_spin)
        form.addRow("Frame", self.frame_combo)
        form.addRow("Start datetime (UTC)", self.start_datetime)
        form.addRow("Carrington rotation", self.cr_num_spin)
        form.addRow("Carrington lon init", self.cr_lon_init_spin)
        form.addRow("Equivalent", self.start_equivalent_label)

        model_box.setLayout(form)
        layout.addWidget(model_box)

        self.setLayout(layout)
        self._sync_from_datetime()
        self._on_1d_toggled(self.one_d_toggle.isChecked())

    def _on_1d_toggled(self, enabled: bool):
        """Update controls when switching between 1D and multi-longitude runs."""
        self.lon_min_spin.setEnabled(not enabled)
        self.lon_max_spin.setEnabled(not enabled)
        if enabled:
            self.frame_combo.setCurrentText("synodic")

    def _sync_from_datetime(self):
        """Update Carrington fields and equivalent text from datetime input."""
        dt = self.start_datetime.dateTime().toPyDateTime()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        cr_num, cr_lon_init = sin.datetime2surfinputs(dt)

        self.cr_num_spin.blockSignals(True)
        self.cr_lon_init_spin.blockSignals(True)
        self.cr_num_spin.setValue(int(cr_num))
        self.cr_lon_init_spin.setValue(cr_lon_init.to(u.deg).value)
        self.cr_num_spin.blockSignals(False)
        self.cr_lon_init_spin.blockSignals(False)

        self.start_equivalent_label.setText(
            f"CR={int(cr_num)}, lon_init={cr_lon_init.to(u.deg).value:.2f} deg"
        )
        self.start_datetime_updated.emit(dt)

    def _sync_from_carrington(self):
        """Update datetime and equivalent text from Carrington inputs."""
        cr_num = self.cr_num_spin.value()
        cr_lon_deg = self.cr_lon_init_spin.value()
        cr_frac = cr_num + ((360.0 - cr_lon_deg) / 360.0)
        start_time = sun.carrington_rotation_time(cr_frac).to_datetime()
        if isinstance(start_time, datetime.datetime):
            text = start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            text = str(start_time)
            start_time = datetime.datetime.utcnow()

        self.start_datetime.blockSignals(True)
        self.start_datetime.setDateTime(QDateTime(start_time))
        self.start_datetime.blockSignals(False)
        self.start_equivalent_label.setText(f"Datetime={text}")
        self.start_datetime_updated.emit(start_time)

    def get_state(self):
        """Return current values as a plain dictionary for code generation."""
        return {
            "rmin": self.rmin_spin.value(),
            "rmax": self.rmax_spin.value(),
            "lon_min": self.lon_min_spin.value(),
            "lon_max": self.lon_max_spin.value(),
            "latitude": self.latitude_spin.value(),
            "is_1d": self.one_d_toggle.isChecked(),
            "frame": self.frame_combo.currentText(),
            "start_datetime": self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "cr_num": self.cr_num_spin.value(),
            "cr_lon_init_deg": self.cr_lon_init_spin.value(),
        }


class PlaceholderTab(QWidget):
    """Simple placeholder panel for tabs not yet implemented."""

    def __init__(self, title: str):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel(f"{title} tab coming soon."))
        self.setLayout(layout)


class VisualisationTab(QWidget):
    """Tab containing standard post-run plotting controls."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.map_box = QGroupBox("2D Map (sa.plot)")
        map_form = QFormLayout()
        self.map_time_spin = QDoubleSpinBox()
        self.map_time_spin.setRange(0.0, 1000.0)
        self.map_time_spin.setSingleStep(0.1)
        self.map_time_spin.setValue(1.5)
        self.map_time_spin.setSuffix(" day")
        self.plot_map_button = QPushButton("Plot 2D Map")
        map_form.addRow("Time", self.map_time_spin)
        map_form.addRow(self.plot_map_button)
        self.map_box.setLayout(map_form)

        radial_box = QGroupBox("Radial Profile (sa.plot_radial)")
        radial_form = QFormLayout()
        self.radial_time_spin = QDoubleSpinBox()
        self.radial_time_spin.setRange(0.0, 1000.0)
        self.radial_time_spin.setSingleStep(0.1)
        self.radial_time_spin.setValue(1.5)
        self.radial_time_spin.setSuffix(" day")

        self.radial_lon_spin = QDoubleSpinBox()
        self.radial_lon_spin.setRange(-360.0, 360.0)
        self.radial_lon_spin.setSingleStep(1.0)
        self.radial_lon_spin.setValue(0.0)
        self.radial_lon_spin.setSuffix(" deg")

        self.plot_radial_button = QPushButton("Plot Radial Profile")
        radial_form.addRow("Time", self.radial_time_spin)
        radial_form.addRow("Longitude", self.radial_lon_spin)
        radial_form.addRow(self.plot_radial_button)
        radial_box.setLayout(radial_form)

        ts_box = QGroupBox("Time Series (sa.plot_timeseries)")
        ts_form = QFormLayout()
        self.ts_radius_spin = QDoubleSpinBox()
        self.ts_radius_spin.setRange(0.1, 10.0)
        self.ts_radius_spin.setSingleStep(0.1)
        self.ts_radius_spin.setValue(1.0)
        self.ts_radius_spin.setSuffix(" AU")

        self.ts_lon_spin = QDoubleSpinBox()
        self.ts_lon_spin.setRange(-360.0, 360.0)
        self.ts_lon_spin.setSingleStep(1.0)
        self.ts_lon_spin.setValue(0.0)
        self.ts_lon_spin.setSuffix(" deg")

        self.plot_timeseries_button = QPushButton("Plot Time Series")
        ts_form.addRow("Radius", self.ts_radius_spin)
        ts_form.addRow("Longitude", self.ts_lon_spin)
        ts_form.addRow(self.plot_timeseries_button)
        ts_box.setLayout(ts_form)

        layout.addWidget(self.map_box)
        layout.addWidget(radial_box)
        layout.addWidget(ts_box)
        self.setLayout(layout)

    def set_1d_mode(self, enabled: bool):
        """Disable 2D map controls when the model is configured for 1D runs."""
        self.map_box.setEnabled(not enabled)


class CmeTab(QWidget):
    """Tab for creating and managing ConeCME entries for model runs."""

    def __init__(self):
        super().__init__()
        self._cmes = []
        self.model_start_datetime = datetime.datetime.utcnow()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        add_box = QGroupBox("Add Cone CME")
        add_form = QFormLayout()

        self.cme_lon_spin = QDoubleSpinBox()
        self.cme_lon_spin.setRange(-360.0, 360.0)
        self.cme_lon_spin.setSingleStep(1.0)
        self.cme_lon_spin.setValue(0.0)
        self.cme_lon_spin.setSuffix(" deg")

        self.cme_lat_spin = QDoubleSpinBox()
        self.cme_lat_spin.setRange(-90.0, 90.0)
        self.cme_lat_spin.setSingleStep(1.0)
        self.cme_lat_spin.setValue(0.0)
        self.cme_lat_spin.setSuffix(" deg")

        self.cme_speed_spin = QDoubleSpinBox()
        self.cme_speed_spin.setRange(100.0, 4000.0)
        self.cme_speed_spin.setSingleStep(10.0)
        self.cme_speed_spin.setValue(800.0)
        self.cme_speed_spin.setSuffix(" km/s")

        self.cme_width_spin = QDoubleSpinBox()
        self.cme_width_spin.setRange(1.0, 180.0)
        self.cme_width_spin.setSingleStep(1.0)
        self.cme_width_spin.setValue(60.0)
        self.cme_width_spin.setSuffix(" deg")

        self.cme_launch_spin = QDoubleSpinBox()
        self.cme_launch_spin.setRange(-100.0, 100.0)
        self.cme_launch_spin.setSingleStep(0.1)
        self.cme_launch_spin.setValue(0.5)
        self.cme_launch_spin.setSuffix(" day")

        self.cme_launch_datetime = QDateTimeEdit()
        self.cme_launch_datetime.setCalendarPopup(True)
        self.cme_launch_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.cme_launch_datetime.setTimeSpec(Qt.TimeSpec.UTC)
        self.cme_launch_datetime.setDateTime(QDateTime.currentDateTimeUtc())

        self.cme_launch_spin.valueChanged.connect(self._sync_launch_datetime_from_day)
        self.cme_launch_datetime.dateTimeChanged.connect(self._sync_launch_day_from_datetime)

        self.cme_thickness_spin = QDoubleSpinBox()
        self.cme_thickness_spin.setRange(0.0, 50.0)
        self.cme_thickness_spin.setSingleStep(0.5)
        self.cme_thickness_spin.setValue(5.0)
        self.cme_thickness_spin.setSuffix(" Rs")

        self.cme_initial_height_spin = QDoubleSpinBox()
        self.cme_initial_height_spin.setRange(1.0, 100.0)
        self.cme_initial_height_spin.setSingleStep(0.5)
        self.cme_initial_height_spin.setValue(21.5)
        self.cme_initial_height_spin.setSuffix(" Rs")

        self.cme_expansion_toggle = QCheckBox("Expansion")
        self.cme_fixed_duration_toggle = QCheckBox("Fixed duration")
        self.cme_fixed_duration_toggle.setChecked(True)

        self.cme_fixed_duration_hours_spin = QDoubleSpinBox()
        self.cme_fixed_duration_hours_spin.setRange(0.1, 240.0)
        self.cme_fixed_duration_hours_spin.setSingleStep(0.5)
        self.cme_fixed_duration_hours_spin.setValue(12.0)
        self.cme_fixed_duration_hours_spin.setSuffix(" hr")

        self.profile_type_combo = QComboBox()
        self.profile_type_combo.addItems(["square", "sinusoidal"])

        self.density_fraction_spin = QDoubleSpinBox()
        self.density_fraction_spin.setRange(0.01, 100.0)
        self.density_fraction_spin.setSingleStep(0.1)
        self.density_fraction_spin.setValue(1.0)

        self.temperature_fraction_spin = QDoubleSpinBox()
        self.temperature_fraction_spin.setRange(0.01, 100.0)
        self.temperature_fraction_spin.setSingleStep(0.1)
        self.temperature_fraction_spin.setValue(1.0)

        self.add_cme_button = QPushButton("Add CME")
        self.add_cme_button.clicked.connect(self.add_cme)

        add_form.addRow("HEEQ longitude", self.cme_lon_spin)
        add_form.addRow("HEEQ latitude", self.cme_lat_spin)
        add_form.addRow("Speed", self.cme_speed_spin)
        add_form.addRow("Width", self.cme_width_spin)
        add_form.addRow("Launch time from model start", self.cme_launch_spin)
        add_form.addRow("Launch datetime (UTC)", self.cme_launch_datetime)
        add_form.addRow("Thickness", self.cme_thickness_spin)
        add_form.addRow("Initial height", self.cme_initial_height_spin)
        add_form.addRow("Expansion", self.cme_expansion_toggle)
        add_form.addRow("Fixed duration", self.cme_fixed_duration_toggle)
        add_form.addRow("Fixed duration value", self.cme_fixed_duration_hours_spin)
        add_form.addRow("Profile type", self.profile_type_combo)
        add_form.addRow("Density fraction", self.density_fraction_spin)
        add_form.addRow("Temperature fraction", self.temperature_fraction_spin)
        add_form.addRow(self.add_cme_button)
        add_box.setLayout(add_form)

        list_box = QGroupBox("CME List")
        list_layout = QVBoxLayout()
        self.cme_list_widget = QListWidget()

        button_row = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected_cme)
        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self.clear_cmes)
        button_row.addWidget(self.remove_selected_button)
        button_row.addWidget(self.clear_all_button)

        list_layout.addWidget(self.cme_list_widget)
        list_layout.addLayout(button_row)
        list_box.setLayout(list_layout)

        layout.addWidget(add_box)
        layout.addWidget(list_box)
        self.setLayout(layout)
        self._sync_launch_datetime_from_day()

    def set_model_start_datetime(self, model_start: object):
        """Update model start reference and refresh launch datetime from day offset."""
        if isinstance(model_start, datetime.datetime):
            self.model_start_datetime = model_start.replace(tzinfo=None)
            self._sync_launch_datetime_from_day()

    def _sync_launch_datetime_from_day(self):
        """Sync CME launch datetime display from relative launch day input."""
        launch_dt = self.model_start_datetime + datetime.timedelta(days=self.cme_launch_spin.value())
        self.cme_launch_datetime.blockSignals(True)
        self.cme_launch_datetime.setDateTime(QDateTime(launch_dt))
        self.cme_launch_datetime.blockSignals(False)

    def _sync_launch_day_from_datetime(self):
        """Sync relative launch day from CME launch datetime input."""
        launch_dt = self.cme_launch_datetime.dateTime().toPyDateTime()
        if launch_dt.tzinfo is not None:
            launch_dt = launch_dt.replace(tzinfo=None)
        delta_days = (launch_dt - self.model_start_datetime).total_seconds() / 86400.0
        self.cme_launch_spin.blockSignals(True)
        self.cme_launch_spin.setValue(delta_days)
        self.cme_launch_spin.blockSignals(False)

    def _refresh_cme_list(self):
        """Refresh visible CME list from internal state."""
        self.cme_list_widget.clear()
        for idx, cme in enumerate(self._cmes, start=1):
            line = (
                f"{idx:02d}: t={cme['t_launch_day']} day, lon={cme['longitude']} deg, "
                f"lat={cme['latitude']} deg, v={cme['speed']} km/s, width={cme['width']} deg, "
                f"dt={cme['t_launch_datetime']} UTC"
            )
            self.cme_list_widget.addItem(line)

    def add_cme(self):
        """Add a CME entry with current control values."""
        self._cmes.append(
            {
                "longitude": self.cme_lon_spin.value(),
                "latitude": self.cme_lat_spin.value(),
                "speed": self.cme_speed_spin.value(),
                "width": self.cme_width_spin.value(),
                "t_launch_day": self.cme_launch_spin.value(),
                "t_launch_datetime": self.cme_launch_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                "thickness_rs": self.cme_thickness_spin.value(),
                "initial_height_rs": self.cme_initial_height_spin.value(),
                "cme_expansion": self.cme_expansion_toggle.isChecked(),
                "cme_fixed_duration": self.cme_fixed_duration_toggle.isChecked(),
                "fixed_duration_hr": self.cme_fixed_duration_hours_spin.value(),
                "profile_type": self.profile_type_combo.currentText(),
                "density_fraction": self.density_fraction_spin.value(),
                "temperature_fraction": self.temperature_fraction_spin.value(),
            }
        )
        self._refresh_cme_list()

    def remove_selected_cme(self):
        """Remove the currently selected CME entry, if any."""
        row = self.cme_list_widget.currentRow()
        if row >= 0:
            self._cmes.pop(row)
            self._refresh_cme_list()

    def clear_cmes(self):
        """Remove all CME entries."""
        self._cmes.clear()
        self._refresh_cme_list()

    def get_cmes(self):
        """Return CME entries as plain dictionaries for code generation."""
        return list(self._cmes)


class CodeDialog(QDialog):
    """Window to display generated SURF code from current GUI state."""

    def __init__(self, code_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated SURF Code")
        self.resize(900, 650)

        layout = QVBoxLayout()
        code_box = QTextEdit()
        code_box.setReadOnly(True)
        code_box.setPlainText(code_text)
        layout.addWidget(code_box)
        self.setLayout(layout)


class TerminalOutputDialog(QDialog):
    """Window to display captured run output and tracebacks."""

    def __init__(self, output_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SURF Terminal Output")
        self.resize(900, 650)

        layout = QVBoxLayout()
        output_box = QTextEdit()
        output_box.setReadOnly(True)
        output_box.setPlainText(output_text)
        layout.addWidget(output_box)
        self.setLayout(layout)


class SurfRunWorker(QObject):
    """Background worker to execute generated code without freezing the UI."""

    finished = pyqtSignal(bool, str, str, object)

    def __init__(self, code_text: str):
        super().__init__()
        self.code_text = code_text

    def run(self):
        """Execute generated code and emit success/error status."""
        output_stream = StringIO()
        model_obj = None
        try:
            exec_globals = {}
            with redirect_stdout(output_stream), redirect_stderr(output_stream):
                exec(self.code_text, exec_globals)
            model_obj = exec_globals.get("model")
            self.finished.emit(
                True,
                "SURF run completed successfully.",
                output_stream.getvalue(),
                model_obj,
            )
        except Exception:
            error = traceback.format_exc()
            output_stream.write(error)
            self.finished.emit(False, "SURF run failed.", output_stream.getvalue(), None)


class SurfMainWindow(QMainWindow):
    """Main SURF GUI window with tabbed workflow sections."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SURF GUI")
        self.resize(900, 600)

        self.run_thread = None
        self.run_worker = None
        self.last_terminal_output = "No run output available yet."
        self.last_model = None

        central = QWidget()
        root_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.model_tab = ModelParametersTab()
        self.ambient_tab = PlaceholderTab("Ambient Solar Wind")
        self.cme_tab = CmeTab()
        self.visualisation_tab = VisualisationTab()

        self.tabs.addTab(self.model_tab, "Model Parameters")
        self.tabs.addTab(self.ambient_tab, "Ambient Solar Wind")
        self.tabs.addTab(self.cme_tab, "CMEs")
        self.tabs.addTab(self.visualisation_tab, "Visualisation")
        self.tabs.setTabEnabled(3, False)

        self.visualisation_tab.plot_map_button.clicked.connect(self.plot_map)
        self.visualisation_tab.plot_radial_button.clicked.connect(self.plot_radial)
        self.visualisation_tab.plot_timeseries_button.clicked.connect(self.plot_timeseries)
        self.model_tab.one_d_toggle.toggled.connect(self._on_1d_mode_changed)
        self.model_tab.start_datetime_updated.connect(self.cme_tab.set_model_start_datetime)
        self._on_1d_mode_changed(self.model_tab.one_d_toggle.isChecked())
        self.cme_tab.set_model_start_datetime(self.model_tab.start_datetime.dateTime().toPyDateTime())
        root_layout.addWidget(self.tabs)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 8, 0, 0)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)

        self.show_code_button = QPushButton("Show Code")
        self.show_code_button.clicked.connect(self.show_generated_code)
        footer.addWidget(self.show_code_button)

        self.show_output_button = QPushButton("Show Terminal Output")
        self.show_output_button.clicked.connect(self.show_terminal_output)
        footer.addWidget(self.show_output_button)

        self.run_button = QPushButton("Run SURF")
        self.run_button.clicked.connect(self.run_surf)
        self._set_run_button_idle_style()
        footer.addWidget(self.run_button)

        root_layout.addLayout(footer)
        central.setLayout(root_layout)
        self.setCentralWidget(central)

    def _build_generated_code(self):
        """Create a runnable Python script from current GUI state."""
        state = self.model_tab.get_state()
        cmes = self.cme_tab.get_cmes()
        lines = [
            "import numpy as np",
            "import astropy.units as u",
            "import surf.surf as s",
            "",
            "# Generated by SURF GUI",
            f"rmin = {state['rmin']} * u.solRad",
            f"rmax = {state['rmax']} * u.solRad",
            f"lon_start = {state['lon_min']} * u.deg",
            f"lon_stop = {state['lon_max']} * u.deg",
            f"latitude = {state['latitude']} * u.deg",
            f"frame = {state['frame']!r}",
            f"cr_num = {state['cr_num']}",
            f"cr_lon_init = {state['cr_lon_init_deg']} * u.deg",
            "",
            "# Use example-style setup with an explicit boundary profile.",
            "v_boundary = np.ones(128) * 400 * (u.km / u.s)",
            "",
            "model = s.SURF(",
            "    v_boundary=v_boundary,",
            "    cr_num=cr_num,",
            "    cr_lon_init=cr_lon_init,",
            "    r_min=rmin,",
            "    r_max=rmax,",
        ]

        if state["is_1d"]:
            lines.append("    lon_out=0.0 * u.deg,")
        else:
            lines.extend(
                [
                    "    lon_start=lon_start,",
                    "    lon_stop=lon_stop,",
                ]
            )

        lines.extend(
            [
                "    latitude=latitude,",
                "    frame=frame,",
                "    simtime=5 * u.day,",
                "    dt_scale=4,",
                ")",
                "",
                "cme_list = []",
            ]
        )

        for idx, cme in enumerate(cmes):
            lines.extend(
                [
                    (
                        f"cme_{idx} = s.ConeCME("
                        f"t_launch={cme['t_launch_day']}*u.day, "
                        f"longitude={cme['longitude']}*u.deg, "
                        f"latitude={cme['latitude']}*u.deg, "
                        f"width={cme['width']}*u.deg, "
                        f"v={cme['speed']}*(u.km/u.s), "
                        f"thickness={cme['thickness_rs']}*u.solRad, "
                        f"initial_height={cme['initial_height_rs']}*u.solRad, "
                        f"cme_expansion={cme['cme_expansion']}, "
                        f"cme_fixed_duration={cme['cme_fixed_duration']}, "
                        f"fixed_duration={cme['fixed_duration_hr']}*u.hour, "
                        f"density_fraction={cme['density_fraction']}, "
                        f"temperature_fraction={cme['temperature_fraction']}, "
                        f"profile_type={cme['profile_type']!r})"
                    ),
                    f"cme_list.append(cme_{idx})",
                ]
            )

        lines.append("model.solve(cme_list)")

        return "\n".join(lines) + "\n"

    def _set_run_button_idle_style(self):
        """Apply neutral style used when idle or before first run."""
        self.run_button.setStyleSheet("")

    def _set_run_button_running_style(self):
        """Apply running style (red) while SURF job executes."""
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )

    def _set_run_button_success_style(self):
        """Apply completion style (green) after successful run."""
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #228b22; color: white; font-weight: 600; }"
        )

    def _set_run_button_failed_style(self):
        """Apply failure style to indicate run ended with an exception."""
        self.run_button.setStyleSheet(
            "QPushButton { background-color: #8b4513; color: white; font-weight: 600; }"
        )

    def _on_1d_mode_changed(self, enabled: bool):
        """Propagate 1D mode UI state to visualisation controls."""
        self.visualisation_tab.set_1d_mode(enabled)

    def show_generated_code(self):
        """Show a text window with generated script based on current GUI state."""
        code_text = self._build_generated_code()
        dialog = CodeDialog(code_text, self)
        dialog.exec()

    def show_terminal_output(self):
        """Show captured output from the most recent SURF run in a separate window."""
        dialog = TerminalOutputDialog(self.last_terminal_output, self)
        dialog.exec()

    def run_surf(self):
        """Execute generated SURF code and update UI state accordingly."""
        code_text = self._build_generated_code()
        self.status_label.setText("Running SURF...")
        self.run_button.setEnabled(False)
        self.show_code_button.setEnabled(False)
        self.show_output_button.setEnabled(False)
        self._set_run_button_running_style()

        self.run_thread = QThread(self)
        self.run_worker = SurfRunWorker(code_text)
        self.run_worker.moveToThread(self.run_thread)

        self.run_thread.started.connect(self.run_worker.run)
        self.run_worker.finished.connect(self._on_run_finished)
        self.run_worker.finished.connect(self.run_thread.quit)
        self.run_worker.finished.connect(self.run_worker.deleteLater)
        self.run_thread.finished.connect(self.run_thread.deleteLater)
        self.run_thread.start()

    def _append_terminal_output(self, text: str):
        """Append text to the captured terminal output buffer."""
        if not text:
            return
        if self.last_terminal_output and self.last_terminal_output != "No run output available yet.":
            self.last_terminal_output += "\n\n" + text
        else:
            self.last_terminal_output = text

    def _on_plot_failed(self, action: str):
        """Record plotting traceback and notify user to inspect output dialog."""
        error = traceback.format_exc()
        self._append_terminal_output(error)
        self.status_label.setText(f"{action} failed. Open 'Show Terminal Output' for details.")

    def _on_plot_succeeded(self, action: str):
        """Update status after a successful plotting call."""
        self.status_label.setText(f"{action} generated.")

    def plot_map(self):
        """Generate a 2D map plot matching notebook sa.plot usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            plot_time = self.visualisation_tab.map_time_spin.value() * u.day
            sa.plot(self.last_model, plot_time)
            plt.show()
            self._on_plot_succeeded("2D map")
        except Exception:
            self._on_plot_failed("2D map")

    def plot_radial(self):
        """Generate a radial profile plot matching notebook sa.plot_radial usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            plot_time = self.visualisation_tab.radial_time_spin.value() * u.day
            lon = self.visualisation_tab.radial_lon_spin.value() * u.deg
            sa.plot_radial(self.last_model, plot_time, lon=lon)
            plt.show()
            self._on_plot_succeeded("Radial profile")
        except Exception:
            self._on_plot_failed("Radial profile")

    def plot_timeseries(self):
        """Generate a time series plot matching notebook sa.plot_timeseries usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            radius = self.visualisation_tab.ts_radius_spin.value() * u.AU
            lon = self.visualisation_tab.ts_lon_spin.value() * u.deg
            sa.plot_timeseries(self.last_model, radius, lon=lon)
            plt.show()
            self._on_plot_succeeded("Time series")
        except Exception:
            self._on_plot_failed("Time series")

    def _on_run_finished(self, success: bool, message: str, terminal_output: str, model_obj):
        """Handle SURF completion state and update UI styling/availability."""
        self.run_button.setEnabled(True)
        self.show_code_button.setEnabled(True)
        self.show_output_button.setEnabled(True)
        self.last_terminal_output = terminal_output or "(No terminal output captured.)"

        if success:
            self._set_run_button_success_style()
            self.status_label.setText(message)
            self.last_model = model_obj
            self.tabs.setTabEnabled(3, True)
        else:
            self._set_run_button_failed_style()
            self.status_label.setText(message + " Open 'Show Terminal Output' for details.")


def main():
    """Launch the SURF GUI application."""
    app = QApplication(sys.argv)
    window = SurfMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
