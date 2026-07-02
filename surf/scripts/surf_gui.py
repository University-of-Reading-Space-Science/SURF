"""PyQt GUI for configuring and running SURF workflows."""

import datetime
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from PyQt6.QtCore import QDateTime, QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from sunpy.coordinates import sun

import surf.surf_analysis as sa
import surf.surf_inputs as sin


EXAMPLE_INPUTS_DIR = Path(__file__).resolve().parents[1] / "data" / "example_inputs"


def parse_wsa_start_time(filepath: Path):
    """Extract WSA map time from FITS metadata when available, else filename."""
    filepath = Path(filepath)

    try:
        header = fits.getheader(filepath)
        for key in ("DATE-OBS", "DATE_OBS", "DATE", "MAPDATE"):
            if key in header:
                value = str(header[key]).strip()
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        return datetime.datetime.strptime(value[:19], fmt)
                    except ValueError:
                        continue
    except Exception:
        pass

    name = filepath.name
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})Z", name)
    if match:
        return datetime.datetime.strptime(
            f"{match.group(1)}T{match.group(2)}", "%Y-%m-%dT%H"
        )

    match = re.search(r"(\d{8})(\d{2})", name)
    if match:
        return datetime.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H")

    return None


def parse_cortom_start_time(filepath: Path):
    """Extract CorTom map time from filename."""
    filepath = Path(filepath)
    match = re.search(r"(\d{14})", filepath.name)
    if match:
        return datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


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

        self.simtime_spin = QDoubleSpinBox()
        self.simtime_spin.setRange(0.1, 365.0)
        self.simtime_spin.setSingleStep(0.5)
        self.simtime_spin.setValue(5.0)
        self.simtime_spin.setSuffix(" day")

        self.include_bpol_toggle = QCheckBox("Include bpol in run")
        self.include_bpol_toggle.setChecked(False)

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

        self.start_datetime.dateTimeChanged.connect(self._sync_from_datetime)
        self.cr_num_spin.valueChanged.connect(self._sync_from_carrington)
        self.cr_lon_init_spin.valueChanged.connect(self._sync_from_carrington)

        r_bounds_row = QWidget()
        r_bounds_layout = QHBoxLayout()
        r_bounds_layout.setContentsMargins(0, 0, 0, 0)
        r_bounds_layout.addWidget(QLabel("rmin"))
        r_bounds_layout.addWidget(self.rmin_spin)
        r_bounds_layout.addWidget(QLabel("rmax"))
        r_bounds_layout.addWidget(self.rmax_spin)
        r_bounds_row.setLayout(r_bounds_layout)

        run_mode_row = QWidget()
        run_mode_layout = QHBoxLayout()
        run_mode_layout.setContentsMargins(0, 0, 0, 0)
        run_mode_layout.addWidget(self.one_d_toggle)
        run_mode_layout.addSpacing(12)
        run_mode_layout.addWidget(QLabel("Long range"))
        run_mode_layout.addWidget(self.lon_min_spin)
        run_mode_layout.addWidget(QLabel("to"))
        run_mode_layout.addWidget(self.lon_max_spin)
        run_mode_row.setLayout(run_mode_layout)

        start_carr_row = QWidget()
        start_carr_layout = QHBoxLayout()
        start_carr_layout.setContentsMargins(0, 0, 0, 0)
        start_carr_layout.addWidget(self.start_datetime, 2)
        start_carr_layout.addWidget(QLabel("CR"))
        start_carr_layout.addWidget(self.cr_num_spin)
        start_carr_layout.addWidget(QLabel("Earth Carr lon at start"))
        start_carr_layout.addWidget(self.cr_lon_init_spin)
        start_carr_row.setLayout(start_carr_layout)

        form.addRow("Radial bounds", r_bounds_row)
        form.addRow("Run model", run_mode_row)
        form.addRow("Latitude", self.latitude_spin)
        form.addRow("Frame", self.frame_combo)
        form.addRow("Run time", self.simtime_spin)
        form.addRow("Magnetic boundary", self.include_bpol_toggle)
        form.addRow("Start / Carrington", start_carr_row)

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
        """Update Carrington fields from datetime input."""
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
        self.start_datetime_updated.emit(dt)

    def _sync_from_carrington(self):
        """Update datetime from Carrington inputs."""
        cr_num = self.cr_num_spin.value()
        cr_lon_deg = self.cr_lon_init_spin.value()
        cr_frac = cr_num + ((360.0 - cr_lon_deg) / 360.0)
        start_time = sun.carrington_rotation_time(cr_frac).to_datetime()
        if not isinstance(start_time, datetime.datetime):
            start_time = datetime.datetime.utcnow()

        self.start_datetime.blockSignals(True)
        self.start_datetime.setDateTime(QDateTime(start_time))
        self.start_datetime.blockSignals(False)
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
            "simtime_days": self.simtime_spin.value(),
            "include_bpol": self.include_bpol_toggle.isChecked(),
            "start_datetime": self.start_datetime.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "cr_num": self.cr_num_spin.value(),
            "cr_lon_init_deg": self.cr_lon_init_spin.value(),
        }

    def set_start_datetime(self, dt: datetime.datetime):
        """Update the run start time from an external source such as a selected file."""
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.start_datetime.setDateTime(QDateTime(dt))


class PlaceholderTab(QWidget):
    """Simple placeholder panel for tabs not yet implemented."""

    def __init__(self, title: str):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel(f"{title} tab coming soon."))
        self.setLayout(layout)


class SourceHighlightTabBar(QTabBar):
    """Tab bar that highlights one tab with a green background and black text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighted_index = -1

    def set_highlighted_index(self, index: int):
        """Set the tab index that should be highlighted for run-source context."""
        if self.highlighted_index != index:
            self.highlighted_index = index
            self.update()

    def paintEvent(self, event):
        """Paint default tabs, then overlay the source-highlighted tab."""
        super().paintEvent(event)

        if self.highlighted_index < 0 or self.highlighted_index >= self.count():
            return

        rect = self.tabRect(self.highlighted_index).adjusted(1, 1, -1, -1)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(rect, QColor("#7fd37f"))
        painter.setPen(QColor("#2f7f2f"))
        painter.drawRect(rect)
        painter.setPen(QColor("#000000"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.tabText(self.highlighted_index))
        painter.end()


class UserSpecifiedAmbientTab(QWidget):
    """Controls for a simple user-defined ambient boundary profile."""

    def __init__(self):
        super().__init__()

        self.num_points = 128
        self.min_speed_kms = 250
        self.max_speed_kms = 750
        # Broader coupling kernel to affect a wider longitude range while dragging.
        self._smoothing_weights = [
            1.0,
            0.92,
            0.84,
            0.76,
            0.68,
            0.6,
            0.52,
            0.44,
            0.36,
            0.28,
            0.2,
            0.12,
        ]
        self._updating_sliders = False
        self.speed_values = np.ones(self.num_points) * 400.0
        self.slider_height_px = 100
        self.slider_col_width_px = 6
        self.sliders = []

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("User-specified boundary")
        box_layout = QVBoxLayout()

        info = QLabel(
            "Set solar-wind speed by Carrington longitude (128 points). "
            "Dragging one slider also adjusts nearby longitudes to reduce discontinuities. "
            "0 and 360 boundaries are periodic."
        )
        info.setWordWrap(True)

        button_row = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_profile)

        self.smooth_button = QPushButton("Smooth")
        self.smooth_button.clicked.connect(self._smooth_profile)

        self.randomise_button = QPushButton("Randomise")
        self.randomise_button.clicked.connect(self._randomise_profile)

        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.smooth_button)
        button_layout.addWidget(self.randomise_button)
        button_layout.addStretch(1)
        button_row.setLayout(button_layout)

        y_axis_title = QLabel("Speed [km/s]")
        y_axis_title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        profile_row = QWidget()
        profile_layout = QHBoxLayout()
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setSpacing(6)
        profile_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        y_axis_col = QWidget()
        y_axis_col_layout = QVBoxLayout()
        y_axis_col_layout.setContentsMargins(0, 0, 0, 0)
        y_axis_col_layout.setSpacing(0)
        y_axis_col_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        y_axis_ticks = QWidget()
        y_axis_ticks.setFixedSize(64, self.slider_height_px)
        for speed in range(self.min_speed_kms, self.max_speed_kms + 1, 100):
            ratio = (self.max_speed_kms - speed) / (self.max_speed_kms - self.min_speed_kms)
            y_pos = int(round(ratio * (self.slider_height_px - 1)))

            tick_text = QLabel(str(speed), y_axis_ticks)
            tick_text.adjustSize()
            text_y = max(0, min(self.slider_height_px - tick_text.height(), y_pos - tick_text.height() // 2))
            tick_text.move(0, text_y)

            tick_mark = QLabel("-", y_axis_ticks)
            tick_mark.adjustSize()
            mark_y = max(0, min(self.slider_height_px - tick_mark.height(), y_pos - tick_mark.height() // 2))
            tick_mark.move(48, mark_y)

        y_axis_col_layout.addWidget(y_axis_ticks)
        y_axis_col.setLayout(y_axis_col_layout)
        y_axis_col.setFixedHeight(self.slider_height_px)

        slider_container = QWidget()
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(4, 4, 4, 4)
        slider_layout.setSpacing(0)

        for idx in range(self.num_points):
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)

            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(self.min_speed_kms, self.max_speed_kms)
            slider.setValue(int(self.speed_values[idx]))
            slider.setFixedHeight(self.slider_height_px)
            slider.setFixedWidth(self.slider_col_width_px)
            slider.valueChanged.connect(lambda value, i=idx: self._on_slider_changed(i, value))

            col.addWidget(slider)

            col_widget = QWidget()
            col_widget.setFixedWidth(self.slider_col_width_px)
            col_widget.setLayout(col)
            slider_layout.addWidget(col_widget)

            self.sliders.append(slider)

        slider_strip_width = self.num_points * self.slider_col_width_px
        slider_container.setFixedWidth(slider_strip_width + 8)

        slider_container.setLayout(slider_layout)

        profile_layout.addWidget(y_axis_col)
        profile_layout.addWidget(slider_container)
        profile_layout.addStretch(1)
        profile_row.setLayout(profile_layout)

        x_axis_row = QWidget()
        x_axis_layout = QHBoxLayout()
        x_axis_layout.setContentsMargins(0, 0, 0, 0)
        x_axis_layout.setSpacing(6)

        x_axis_spacer = QWidget()
        x_axis_spacer.setFixedWidth(64)
        x_axis_layout.addWidget(x_axis_spacer)

        x_axis_ticks = QWidget()
        x_axis_ticks.setFixedSize(slider_strip_width + 8, 18)
        for deg in [0, 90, 180, 270, 360]:
            ratio = deg / 360.0
            x_pos = int(round(4 + ratio * slider_strip_width))
            tick = QLabel(str(deg), x_axis_ticks)
            tick.adjustSize()
            tick_x = max(0, min(x_axis_ticks.width() - tick.width(), x_pos - tick.width() // 2))
            tick.move(tick_x, 0)

        x_axis_layout.addWidget(x_axis_ticks)
        x_axis_layout.addStretch(1)
        x_axis_row.setLayout(x_axis_layout)

        x_axis_title = QLabel("Carrington longitude [deg]")
        x_axis_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        box_layout.addWidget(info)
        box_layout.addWidget(button_row)
        box_layout.addWidget(y_axis_title)
        box_layout.addWidget(profile_row)
        box_layout.addWidget(x_axis_row)
        box_layout.addWidget(x_axis_title)
        box.setLayout(box_layout)

        layout.addWidget(box)
        self.setLayout(layout)
        self._randomise_profile()

    def _wrapped_distance(self, idx_a: int, idx_b: int) -> int:
        """Return circular index distance on the 128-longitude grid."""
        direct = abs(idx_a - idx_b)
        return min(direct, self.num_points - direct)

    def _set_speed_values(self, values: np.ndarray):
        """Apply speed values to state and slider widgets without recursive signal loops."""
        clipped = np.clip(values, self.min_speed_kms, self.max_speed_kms)
        self.speed_values = clipped.astype(float)

        self._updating_sliders = True
        for idx in range(self.num_points):
            slider_value = int(round(self.speed_values[idx]))
            self.speed_values[idx] = slider_value
            self.sliders[idx].setValue(slider_value)
        self._updating_sliders = False

    def _on_slider_changed(self, idx: int, value: int):
        """Update selected slider and nearby longitudes to smooth sharp jumps."""
        if self._updating_sliders:
            return

        old_value = float(self.speed_values[idx])
        delta = float(value) - old_value
        if abs(delta) < 1.0e-9:
            return

        updated = self.speed_values.copy()
        for j in range(self.num_points):
            dist = self._wrapped_distance(idx, j)
            if dist < len(self._smoothing_weights):
                updated[j] += delta * self._smoothing_weights[dist]

        # Ensure the selected index follows the user drag exactly.
        updated[idx] = float(value)
        self._set_speed_values(updated)

    def _reset_profile(self):
        """Reset all longitudes to a nominal 400 km/s profile."""
        self._set_speed_values(np.ones(self.num_points) * 400.0)

    def _smooth_profile(self):
        """Apply periodic smoothing across all longitudes."""
        kernel = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        smoothed = np.zeros_like(self.speed_values, dtype=float)
        center = len(kernel) // 2

        for i in range(self.num_points):
            accum = 0.0
            for k, w in enumerate(kernel):
                j = (i + k - center) % self.num_points
                accum += w * self.speed_values[j]
            smoothed[i] = accum

        self._set_speed_values(smoothed)

    def _randomise_profile(self):
        """Generate a bounded random profile and smooth it once for realism."""
        random_values = np.random.randint(
            self.min_speed_kms,
            self.max_speed_kms + 1,
            size=self.num_points,
        ).astype(float)
        self._set_speed_values(random_values)
        self._smooth_profile()

    def get_state(self):
        """Return current user-specified ambient settings."""
        return {
            "num_points": self.num_points,
            "speed_profile_kms": self.speed_values.tolist(),
        }


class MasAmbientTab(QWidget):
    """Controls for previewing a MAS-derived ambient boundary profile."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.model_inner_boundary_rs = 21.5
        self.model_latitude_deg = 0.0
        self.include_bpol = False

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("HelioMAS boundary profile")
        form = QFormLayout()

        self.cr_spin = QSpinBox()
        self.cr_spin.setRange(1, 10000)
        self.cr_spin.setValue(2000)

        self.decelerate_toggle = QCheckBox("Decelerate to inner boundary")
        self.decelerate_toggle.setChecked(True)

        self.use_map_time_toggle = QCheckBox("Use map time as model start time")
        self.use_map_time_toggle.setChecked(True)
        self.use_map_time_toggle.toggled.connect(self._on_use_map_time_toggled)

        self.plot_button = QPushButton("Extract and Plot Vin")
        self.plot_button.clicked.connect(self.plot_profile)
        self.cr_spin.valueChanged.connect(self._emit_map_start_time_if_enabled)

        form.addRow("Carrington rotation", self.cr_spin)
        form.addRow("", self.decelerate_toggle)
        form.addRow("", self.use_map_time_toggle)
        form.addRow(self.plot_button)
        box.setLayout(form)

        info = QLabel(
            "Extract Vin with surf.surf_inputs.get_MAS_long_profile and plot it as a "
            "function of Carrington longitude."
        )
        info.setWordWrap(True)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)

    def get_state(self):
        """Return current MAS settings."""
        return {
            "cr_num": self.cr_spin.value(),
            "decelerate_to_inner_boundary": self.decelerate_toggle.isChecked(),
        }

    def plot_profile(self):
        """Extract MAS Vin and plot source and mapped profiles against longitude."""
        original_text = self.plot_button.text()
        original_style = self.plot_button.styleSheet()
        self.plot_button.setText("downloading and processing")
        self.plot_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )
        self.plot_button.setEnabled(False)
        QApplication.processEvents()

        try:
            cr_num = self.cr_spin.value()
            latitude = self.model_latitude_deg * u.deg
            map_to_inner = self.decelerate_toggle.isChecked()
            v_orig = sin.get_MAS_long_profile(cr_num, latitude)
            if self.include_bpol:
                b_orig = sin.get_MAS_br_long_profile(cr_num, latitude)
                if len(b_orig) != len(v_orig):
                    b_lon = np.linspace(0.0, 360.0, len(b_orig), endpoint=False)
                    v_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
                    b_orig = np.interp(v_lon, b_lon, np.asarray(b_orig), period=360.0)
                if map_to_inner:
                    mapped = sin.map_v_boundary_inwards(
                        v_orig,
                        30.0 * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                        b_orig=b_orig,
                    )
                    if isinstance(mapped, tuple):
                        v_mapped, b_mapped = mapped
                    else:
                        v_mapped = mapped
                        b_mapped = np.ones(len(v_orig)) * np.nan
                else:
                    v_mapped = v_orig
                    b_mapped = b_orig
            else:
                if map_to_inner:
                    v_mapped = sin.map_v_boundary_inwards(
                        v_orig,
                        30.0 * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                    )
                else:
                    v_mapped = v_orig
            carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)

            if self.include_bpol:
                fig, (ax_v, ax_b) = plt.subplots(2, 1, sharex=True)
            else:
                fig, ax_v = plt.subplots()

            ax_v.plot(
                carr_lon,
                v_orig.to_value(u.km / u.s),
                linewidth=1.5,
                label="Original at 30 Rs",
            )
            ax_v.plot(
                carr_lon,
                v_mapped.to_value(u.km / u.s),
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"Mapped to {self.model_inner_boundary_rs:.1f} Rs"
                    if map_to_inner
                    else "Original (no deceleration mapping)"
                ),
            )
            ax_v.set_xlim(0.0, 360.0)
            ax_v.set_ylabel("Vin [km/s]")
            ax_v.set_title(f"MAS boundary profiles | CR {cr_num} | lat {latitude.value:.1f} deg")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()

            if self.include_bpol:
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_orig),
                    linewidth=1.5,
                    label="Original bpol at 30 Rs",
                )
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_mapped),
                    linewidth=1.5,
                    linestyle="--",
                    label=(
                        f"Mapped bpol to {self.model_inner_boundary_rs:.1f} Rs"
                        if map_to_inner
                        else "Original bpol (no deceleration mapping)"
                    ),
                )
                ax_b.set_xlim(0.0, 360.0)
                ax_b.set_xlabel("Carrington longitude [deg]")
                ax_b.set_ylabel("bpol")
                ax_b.grid(True, alpha=0.3)
                ax_b.legend()
            else:
                ax_v.set_xlabel("Carrington longitude [deg]")

            fig.tight_layout()
            plt.show()
            self.status_message.emit("MAS Vin profile plotted.")
        except Exception:
            self.error_message.emit(traceback.format_exc())
        finally:
            self.plot_button.setEnabled(True)
            self.plot_button.setText(original_text)
            self.plot_button.setStyleSheet(original_style)

    def set_model_inner_boundary(self, rmin_rs: float):
        """Set model inner boundary radius used for profile comparison mapping."""
        self.model_inner_boundary_rs = float(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        """Set model latitude used for ambient profile extraction."""
        self.model_latitude_deg = float(latitude_deg)

    def set_include_bpol(self, include_bpol: bool):
        """Enable/disable bpol extraction in profile preview plots."""
        self.include_bpol = bool(include_bpol)

    def _mas_map_time(self) -> datetime.datetime:
        """Infer MAS map time from the selected Carrington rotation."""
        map_time = sun.carrington_rotation_time(float(self.cr_spin.value())).to_datetime()
        if map_time.tzinfo is not None:
            map_time = map_time.replace(tzinfo=None)
        return map_time

    def _emit_map_start_time_if_enabled(self):
        """Emit MAS map time to update model start time when enabled."""
        if not self.use_map_time_toggle.isChecked():
            return

        map_time = self._mas_map_time()
        self.start_time_selected.emit(map_time)
        self.status_message.emit("Run start time updated from MAS map time.")

    def emit_map_time_if_enabled(self):
        """Public helper to apply the current MAS map time when enabled."""
        self._emit_map_start_time_if_enabled()

    def _on_use_map_time_toggled(self, enabled: bool):
        """Apply map-time start updates only when the option is enabled."""
        if enabled:
            self._emit_map_start_time_if_enabled()


class InSituAmbientTab(QWidget):
    """Controls for OMNI backmapped reconstruction or forecast boundary setup."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("OMNI backmapped boundary")
        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["reconstruction", "forecast"])

        info = QLabel(
            "Uses the model start time and run time to initialize the OMNI-based "
            "SURF setup following Example 27."
        )
        info.setWordWrap(True)

        form.addRow("Mode", self.mode_combo)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)

    def get_state(self):
        """Return current InSitu settings."""
        return {"mode": self.mode_combo.currentText()}


class OmniAmbientTab(QWidget):
    """Controls for Example-20 OMNI-driven time-dependent boundaries."""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("OMNI boundary (Example 20)")
        form = QFormLayout()

        self.use_215_inner_boundary_toggle = QCheckBox(
            "Set inner boundary to 215 Rs and run outwards"
        )
        self.use_215_inner_boundary_toggle.setChecked(True)

        info = QLabel(
            "Build time-dependent boundary conditions directly from OMNI observations "
            "following Example 20."
        )
        info.setWordWrap(True)

        form.addRow("", self.use_215_inner_boundary_toggle)
        box.setLayout(form)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)

    def get_state(self):
        """Return current OMNI settings."""
        return {
            "use_215_inner_boundary": self.use_215_inner_boundary_toggle.isChecked(),
        }


class FileAmbientTab(QWidget):
    """Shared UI for ambient sources that select a single input file."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(
        self,
        title: str,
        filter_text: str,
        parser,
        description: str,
        include_decelerate_option: bool = False,
        include_use_map_time_option: bool = False,
        default_pattern: str = "",
        profile_loader=None,
        br_profile_loader=None,
        source_radius_rs: float | None = None,
    ):
        super().__init__()
        self.filter_text = filter_text
        self.parser = parser
        self.selected_file = ""
        self.include_decelerate_option = include_decelerate_option
        self.include_use_map_time_option = include_use_map_time_option
        self.default_pattern = default_pattern
        self.profile_loader = profile_loader
        self.br_profile_loader = br_profile_loader
        self.source_radius_rs = source_radius_rs
        self.model_inner_boundary_rs = 21.5
        self.model_latitude_deg = 0.0
        self.include_bpol = False
        self.last_parsed_time = None

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox(title)
        form = QFormLayout()

        file_row = QWidget()
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(0, 0, 0, 0)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.select_button = QPushButton("Select file")
        self.select_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.select_button)
        file_row.setLayout(file_layout)

        self.detected_time_label = QLabel("No file selected.")
        self.detected_time_label.setWordWrap(True)

        if self.profile_loader is not None:
            self.plot_button = QPushButton("Extract and Plot Vin")
            self.plot_button.clicked.connect(self.plot_profile)

        if self.include_decelerate_option:
            self.decelerate_toggle = QCheckBox("Decelerate to inner boundary")
            self.decelerate_toggle.setChecked(True)

        if self.include_use_map_time_option:
            self.use_map_time_toggle = QCheckBox("Use map time as model start time")
            self.use_map_time_toggle.setChecked(True)
            self.use_map_time_toggle.toggled.connect(self._on_use_map_time_toggled)

        form.addRow("File", file_row)
        form.addRow("Detected start time", self.detected_time_label)
        if self.include_decelerate_option:
            form.addRow("", self.decelerate_toggle)
        if self.include_use_map_time_option:
            form.addRow("", self.use_map_time_toggle)
        if self.profile_loader is not None:
            form.addRow(self.plot_button)
        box.setLayout(form)

        info = QLabel(description)
        info.setWordWrap(True)

        layout.addWidget(box)
        layout.addWidget(info)
        self.setLayout(layout)
        self._load_default_file()

    def select_file(self):
        """Open a file chooser rooted at the SURF example inputs directory."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select file",
            str(EXAMPLE_INPUTS_DIR),
            self.filter_text,
        )
        if filepath:
            self._apply_selected_file(filepath)

    def _apply_selected_file(self, filepath: str):
        """Apply a selected file path and emit a start-time update when possible."""
        try:
            parsed_time = self.parser(Path(filepath))
            self.selected_file = filepath
            self.file_edit.setText(filepath)
            self.last_parsed_time = parsed_time

            if parsed_time is not None:
                self.detected_time_label.setText(parsed_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
                if self._should_apply_map_time():
                    self.start_time_selected.emit(parsed_time)
                    self.status_message.emit("Run start time updated from selected file map time.")
                else:
                    self.status_message.emit("File selected. Map time not applied to model start.")
            else:
                self.detected_time_label.setText("Could not determine date from metadata or filename.")
                self.status_message.emit("File selected, but no start time could be inferred.")
        except Exception:
            self.error_message.emit(traceback.format_exc())

    def get_state(self):
        """Return selected file path for code generation."""
        state = {"filepath": self.selected_file}
        if self.include_decelerate_option:
            state["decelerate_to_inner_boundary"] = self.decelerate_toggle.isChecked()
        return state

    def _load_default_file(self):
        """Select a default example file when available on the local machine."""
        if not self.default_pattern:
            return

        matches = sorted(EXAMPLE_INPUTS_DIR.glob(self.default_pattern))
        if not matches:
            return

        self._apply_selected_file(str(matches[0]))

    def set_model_inner_boundary(self, rmin_rs: float):
        """Set model inner boundary radius used for profile comparison mapping."""
        self.model_inner_boundary_rs = float(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        """Set model latitude used for ambient profile extraction."""
        self.model_latitude_deg = float(latitude_deg)

    def set_include_bpol(self, include_bpol: bool):
        """Enable/disable bpol extraction in profile preview plots."""
        self.include_bpol = bool(include_bpol)

    def _should_apply_map_time(self) -> bool:
        """Return True when parsed map time should update model start."""
        if not self.include_use_map_time_option:
            return False
        return self.use_map_time_toggle.isChecked()

    def _on_use_map_time_toggled(self, enabled: bool):
        """Apply stored parsed map time when enabled and available."""
        if enabled and self.last_parsed_time is not None:
            self.start_time_selected.emit(self.last_parsed_time)
            self.status_message.emit("Run start time updated from selected file map time.")

    def emit_map_time_if_enabled(self):
        """Public helper to apply parsed map time when this option is enabled."""
        if self.last_parsed_time is None:
            return
        if not self._should_apply_map_time():
            return
        self.start_time_selected.emit(self.last_parsed_time)

    def plot_profile(self):
        """Extract and plot source profile together with mapped-to-rmin profile."""
        if self.profile_loader is None:
            return

        if not self.selected_file:
            self.status_message.emit("Select an input file before plotting.")
            return

        original_text = self.plot_button.text()
        original_style = self.plot_button.styleSheet()
        self.plot_button.setText("downloading and processing")
        self.plot_button.setStyleSheet(
            "QPushButton { background-color: #b22222; color: white; font-weight: 600; }"
        )
        self.plot_button.setEnabled(False)
        QApplication.processEvents()

        try:
            latitude = self.model_latitude_deg * u.deg
            map_to_inner = (
                self.decelerate_toggle.isChecked()
                if self.include_decelerate_option
                else True
            )
            v_orig = self.profile_loader(self.selected_file, latitude)
            source_radius_rs = 21.5 if self.source_radius_rs is None else self.source_radius_rs
            include_bpol_plot = self.include_bpol and (self.br_profile_loader is not None)
            if include_bpol_plot:
                b_orig = self.br_profile_loader(self.selected_file, latitude)
                if map_to_inner:
                    mapped = sin.map_v_boundary_inwards(
                        v_orig,
                        source_radius_rs * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                        b_orig=b_orig,
                    )
                    if isinstance(mapped, tuple):
                        v_mapped, b_mapped = mapped
                    else:
                        v_mapped = mapped
                        b_mapped = np.ones(len(v_orig)) * np.nan
                else:
                    v_mapped = v_orig
                    b_mapped = b_orig
            else:
                if map_to_inner:
                    v_mapped = sin.map_v_boundary_inwards(
                        v_orig,
                        source_radius_rs * u.solRad,
                        self.model_inner_boundary_rs * u.solRad,
                    )
                else:
                    v_mapped = v_orig
            carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)

            if include_bpol_plot:
                fig, (ax_v, ax_b) = plt.subplots(2, 1, sharex=True)
            else:
                fig, ax_v = plt.subplots()

            ax_v.plot(
                carr_lon,
                v_orig.to_value(u.km / u.s),
                linewidth=1.5,
                label=f"Original at {source_radius_rs:.1f} Rs",
            )
            ax_v.plot(
                carr_lon,
                v_mapped.to_value(u.km / u.s),
                linewidth=1.5,
                linestyle="--",
                label=(
                    f"Mapped to {self.model_inner_boundary_rs:.1f} Rs"
                    if map_to_inner
                    else "Original (no deceleration mapping)"
                ),
            )
            ax_v.set_xlim(0.0, 360.0)
            ax_v.set_ylabel("Vin [km/s]")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()

            if include_bpol_plot:
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_orig),
                    linewidth=1.5,
                    label=f"Original bpol at {source_radius_rs:.1f} Rs",
                )
                ax_b.plot(
                    carr_lon,
                    np.asarray(b_mapped),
                    linewidth=1.5,
                    linestyle="--",
                    label=(
                        f"Mapped bpol to {self.model_inner_boundary_rs:.1f} Rs"
                        if map_to_inner
                        else "Original bpol (no deceleration mapping)"
                    ),
                )
                ax_b.set_xlim(0.0, 360.0)
                ax_b.set_xlabel("Carrington longitude [deg]")
                ax_b.set_ylabel("bpol")
                ax_b.grid(True, alpha=0.3)
                ax_b.legend()
            else:
                ax_v.set_xlabel("Carrington longitude [deg]")

            fig.tight_layout()
            plt.show()
            self.status_message.emit("Ambient profile plotted.")
        except Exception:
            self.error_message.emit(traceback.format_exc())
        finally:
            self.plot_button.setEnabled(True)
            self.plot_button.setText(original_text)
            self.plot_button.setStyleSheet(original_style)


class WsaAmbientTab(FileAmbientTab):
    """WSA boundary file selection tab."""

    def __init__(self):
        super().__init__(
            "WSA boundary file",
            "WSA FITS (*.fits)",
            parse_wsa_start_time,
            "Select a WSA FITS file. The run start time is inferred from FITS metadata when "
            "available, otherwise from the filename.",
            include_decelerate_option=True,
            include_use_map_time_option=True,
            default_pattern="**/*.fits",
            profile_loader=sin.get_WSA_long_profile,
            br_profile_loader=sin.get_WSA_br_long_profile,
            source_radius_rs=21.5,
        )


class CorTomAmbientTab(FileAmbientTab):
    """CorTom boundary file selection tab."""

    def __init__(self):
        super().__init__(
            "CorTom boundary file",
            "CorTom DAT (*.dat)",
            parse_cortom_start_time,
            "Select a CorTom DAT file. The run start time is inferred from the filename.",
            include_decelerate_option=True,
            include_use_map_time_option=True,
            default_pattern="**/*.dat",
            profile_loader=sin.get_CorTom_long_profile,
            source_radius_rs=8.0,
        )


class AmbientSolarWindTab(QWidget):
    """Ambient solar wind configuration with source-specific subtabs."""

    status_message = pyqtSignal(str)
    error_message = pyqtSignal(str)
    start_time_selected = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.source_button_group = QButtonGroup(self)
        self.source_button_group.setExclusive(True)
        self.source_buttons = {}

        source_row = QWidget()
        source_layout = QHBoxLayout()
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("Ambient source for run"))

        source_specs = [
            ("User specified", "user_specified"),
            ("MAS", "mas"),
            ("WSA", "wsa"),
            ("InSitu-backmapped", "insitu_backmapped"),
            ("OMNI", "omni"),
            ("CorTom", "cortom"),
        ]
        for label, source_key in source_specs:
            button = QRadioButton(label)
            self.source_button_group.addButton(button)
            self.source_buttons[source_key] = button
            source_layout.addWidget(button)

        source_layout.addStretch(1)
        source_row.setLayout(source_layout)
        self.source_button_group.buttonToggled.connect(self._on_source_selection_changed)

        self.source_tabs = QTabWidget()
        self.source_tabs.setTabBar(SourceHighlightTabBar(self.source_tabs))
        self.user_tab = UserSpecifiedAmbientTab()
        self.mas_tab = MasAmbientTab()
        self.wsa_tab = WsaAmbientTab()
        self.insitu_tab = InSituAmbientTab()
        self.omni_tab = OmniAmbientTab()
        self.cortom_tab = CorTomAmbientTab()

        self.mas_tab.status_message.connect(self.status_message.emit)
        self.mas_tab.error_message.connect(self.error_message.emit)
        self.mas_tab.start_time_selected.connect(self.start_time_selected.emit)
        self.wsa_tab.status_message.connect(self.status_message.emit)
        self.wsa_tab.error_message.connect(self.error_message.emit)
        self.wsa_tab.start_time_selected.connect(self.start_time_selected.emit)
        self.cortom_tab.status_message.connect(self.status_message.emit)
        self.cortom_tab.error_message.connect(self.error_message.emit)
        self.cortom_tab.start_time_selected.connect(self.start_time_selected.emit)

        self.source_tabs.addTab(self.user_tab, "User specified")
        self.source_tabs.addTab(self.mas_tab, "MAS")
        self.source_tabs.addTab(self.wsa_tab, "WSA")
        self.source_tabs.addTab(self.insitu_tab, "InSitu-backmapped")
        self.source_tabs.addTab(self.omni_tab, "OMNI")
        self.source_tabs.addTab(self.cortom_tab, "CorTom")

        self._source_tab_indices = {
            "user_specified": 0,
            "mas": 1,
            "wsa": 2,
            "insitu_backmapped": 3,
            "omni": 4,
            "cortom": 5,
        }

        # Default ambient source for model runs.
        self.source_buttons["wsa"].setChecked(True)

        layout.addWidget(source_row)
        layout.addWidget(self.source_tabs)
        self.setLayout(layout)
        self._update_source_tab_highlight()

    def _on_source_selection_changed(self, _button: QRadioButton, checked: bool):
        """Apply map time when the selected run source changes."""
        if checked:
            self._update_source_tab_highlight()
            self.emit_active_map_time_if_enabled()

    def _update_source_tab_highlight(self):
        """Highlight the tab label for the selected ambient source in green."""
        tab_bar = self.source_tabs.tabBar()
        selected_source = self._selected_source()
        selected_index = self._source_tab_indices.get(selected_source, -1)
        for idx in range(self.source_tabs.count()):
            tab_bar.setTabTextColor(idx, QColor("#000000"))
        if isinstance(tab_bar, SourceHighlightTabBar):
            tab_bar.set_highlighted_index(selected_index)

    def _selected_source(self) -> str:
        """Return the currently selected ambient source key for model runs."""
        for source_key, button in self.source_buttons.items():
            if button.isChecked():
                return source_key
        return "wsa"

    def emit_active_map_time_if_enabled(self):
        """Apply selected-source map time to model start when that source enables it."""
        source = self._selected_source()
        if source == "mas":
            self.mas_tab.emit_map_time_if_enabled()
        elif source == "wsa":
            self.wsa_tab.emit_map_time_if_enabled()
        elif source == "cortom":
            self.cortom_tab.emit_map_time_if_enabled()

    def get_state(self):
        """Return the selected run source and its parameters."""
        state = {"source": self._selected_source()}
        if state["source"] == "user_specified":
            state.update(self.user_tab.get_state())
        elif state["source"] == "mas":
            state.update(self.mas_tab.get_state())
        elif state["source"] == "wsa":
            state.update(self.wsa_tab.get_state())
        elif state["source"] == "insitu_backmapped":
            state.update(self.insitu_tab.get_state())
        elif state["source"] == "omni":
            state.update(self.omni_tab.get_state())
        elif state["source"] == "cortom":
            state.update(self.cortom_tab.get_state())
        return state

    def set_model_inner_boundary(self, rmin_rs: float):
        """Propagate current model inner boundary to source tabs for comparison plots."""
        self.mas_tab.set_model_inner_boundary(rmin_rs)
        self.wsa_tab.set_model_inner_boundary(rmin_rs)
        self.cortom_tab.set_model_inner_boundary(rmin_rs)

    def set_model_latitude(self, latitude_deg: float):
        """Propagate model latitude to source tabs that extract ambient profiles."""
        self.mas_tab.set_model_latitude(latitude_deg)
        self.wsa_tab.set_model_latitude(latitude_deg)
        self.cortom_tab.set_model_latitude(latitude_deg)

    def set_include_bpol(self, include_bpol: bool):
        """Propagate bpol plotting option to relevant source tabs."""
        self.mas_tab.set_include_bpol(include_bpol)
        self.wsa_tab.set_include_bpol(include_bpol)


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

        self.map_minimalplot_toggle = QCheckBox("Minimal plot")
        self.map_plot_hcs_toggle = QCheckBox("Plot HCS")
        self.map_plot_hcs_toggle.setChecked(True)
        self.map_annotate_toggle = QCheckBox("Annotate plot")
        self.map_annotate_toggle.setChecked(True)
        self.map_trace_earth_toggle = QCheckBox("Trace Earth connection (slow)")

        self.map_limit_rmax_toggle = QCheckBox("Limit outer radius")
        self.map_limit_rmax_toggle.toggled.connect(self._on_map_limit_rmax_toggled)
        self.map_rmax_spin = QDoubleSpinBox()
        self.map_rmax_spin.setRange(1.0, 5000.0)
        self.map_rmax_spin.setSingleStep(5.0)
        self.map_rmax_spin.setValue(240.0)
        self.map_rmax_spin.setSuffix(" Rs")
        self.map_rmax_spin.setEnabled(False)

        rmax_row = QWidget()
        rmax_layout = QHBoxLayout()
        rmax_layout.setContentsMargins(0, 0, 0, 0)
        rmax_layout.addWidget(self.map_limit_rmax_toggle)
        rmax_layout.addWidget(self.map_rmax_spin)
        rmax_layout.addStretch(1)
        rmax_row.setLayout(rmax_layout)

        self.plot_map_button = QPushButton("Plot 2D Map")
        map_form.addRow("Time", self.map_time_spin)
        map_form.addRow("", self.map_minimalplot_toggle)
        map_form.addRow("", self.map_plot_hcs_toggle)
        map_form.addRow("", self.map_annotate_toggle)
        map_form.addRow("", self.map_trace_earth_toggle)
        map_form.addRow("", rmax_row)
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

    def _on_map_limit_rmax_toggled(self, enabled: bool):
        """Enable outer-radius spin box only when radius limiting is requested."""
        self.map_rmax_spin.setEnabled(enabled)


class CmeTab(QWidget):
    """Tab for creating and managing ConeCME entries for model runs."""

    def __init__(self):
        super().__init__()
        self._cmes = []
        self.model_start_datetime = datetime.datetime.utcnow()
        self._last_model_start_datetime = None
        self.model_inner_boundary_rs = 21.5
        self.model_run_duration_days = 5.0

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        load_box = QGroupBox("Load Cone File")
        load_form = QFormLayout()

        load_row = QWidget()
        load_row_layout = QHBoxLayout()
        load_row_layout.setContentsMargins(0, 0, 0, 0)

        self.cone_file_edit = QLineEdit()
        self.cone_file_edit.setReadOnly(True)
        self.load_cone_button = QPushButton("Select cone file")
        self.load_cone_button.clicked.connect(self.load_cone_file)

        load_row_layout.addWidget(self.cone_file_edit)
        load_row_layout.addWidget(self.load_cone_button)
        load_row.setLayout(load_row_layout)

        self.load_cone_status_label = QLabel("No cone file loaded.")
        self.load_cone_status_label.setWordWrap(True)

        load_form.addRow("File", load_row)
        load_form.addRow("Status", self.load_cone_status_label)
        load_box.setLayout(load_form)

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
        self.cme_initial_height_spin.valueChanged.connect(self._update_initial_height_style)

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

        self.cme_plasma_mode_combo = QComboBox()
        self.cme_plasma_mode_combo.addItems(["Fraction of ambient", "Absolute values"])
        self.cme_plasma_mode_combo.currentTextChanged.connect(self._on_cme_plasma_mode_changed)

        self.density_fraction_spin = QDoubleSpinBox()
        self.density_fraction_spin.setRange(0.01, 100.0)
        self.density_fraction_spin.setSingleStep(0.1)
        self.density_fraction_spin.setValue(1.0)

        self.temperature_fraction_spin = QDoubleSpinBox()
        self.temperature_fraction_spin.setRange(0.01, 100.0)
        self.temperature_fraction_spin.setSingleStep(0.1)
        self.temperature_fraction_spin.setValue(1.0)

        self.cme_density_spin = QDoubleSpinBox()
        self.cme_density_spin.setRange(0.0, 1.0e6)
        self.cme_density_spin.setDecimals(3)
        self.cme_density_spin.setSingleStep(1.0)
        self.cme_density_spin.setValue(100.0)
        self.cme_density_spin.setSuffix(" p+/cm^3")

        self.cme_temperature_spin = QDoubleSpinBox()
        self.cme_temperature_spin.setRange(1.0, 1.0e8)
        self.cme_temperature_spin.setSingleStep(1000.0)
        self.cme_temperature_spin.setValue(1.0e5)
        self.cme_temperature_spin.setSuffix(" K")

        self.add_cme_button = QPushButton("Add CME")
        self.add_cme_button.clicked.connect(self.add_cme)

        lon_lat_row = QWidget()
        lon_lat_layout = QHBoxLayout()
        lon_lat_layout.setContentsMargins(0, 0, 0, 0)
        lon_lat_layout.addWidget(QLabel("Lon"))
        lon_lat_layout.addWidget(self.cme_lon_spin)
        lon_lat_layout.addWidget(QLabel("Lat"))
        lon_lat_layout.addWidget(self.cme_lat_spin)
        lon_lat_row.setLayout(lon_lat_layout)

        speed_width_row = QWidget()
        speed_width_layout = QHBoxLayout()
        speed_width_layout.setContentsMargins(0, 0, 0, 0)
        speed_width_layout.addWidget(QLabel("Speed"))
        speed_width_layout.addWidget(self.cme_speed_spin)
        speed_width_layout.addWidget(QLabel("Width"))
        speed_width_layout.addWidget(self.cme_width_spin)
        speed_width_row.setLayout(speed_width_layout)

        launch_row = QWidget()
        launch_layout = QHBoxLayout()
        launch_layout.setContentsMargins(0, 0, 0, 0)
        launch_layout.addWidget(self.cme_launch_spin, 1)
        launch_layout.addWidget(QLabel("or"))
        launch_layout.addWidget(self.cme_launch_datetime, 2)
        launch_row.setLayout(launch_layout)

        size_row = QWidget()
        size_layout = QHBoxLayout()
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(QLabel("Thickness"))
        size_layout.addWidget(self.cme_thickness_spin)
        size_layout.addWidget(QLabel("Initial height"))
        size_layout.addWidget(self.cme_initial_height_spin)
        size_row.setLayout(size_layout)

        duration_row = QWidget()
        duration_layout = QHBoxLayout()
        duration_layout.setContentsMargins(0, 0, 0, 0)
        duration_layout.addWidget(self.cme_expansion_toggle)
        duration_layout.addSpacing(12)
        duration_layout.addWidget(self.cme_fixed_duration_toggle)
        duration_layout.addSpacing(12)
        duration_layout.addWidget(QLabel("Duration"))
        duration_layout.addWidget(self.cme_fixed_duration_hours_spin)
        duration_row.setLayout(duration_layout)

        self.fraction_row = QWidget()
        fraction_layout = QHBoxLayout()
        fraction_layout.setContentsMargins(0, 0, 0, 0)
        fraction_layout.addWidget(QLabel("Density"))
        fraction_layout.addWidget(self.density_fraction_spin)
        fraction_layout.addWidget(QLabel("Temperature"))
        fraction_layout.addWidget(self.temperature_fraction_spin)
        self.fraction_row.setLayout(fraction_layout)

        self.absolute_row = QWidget()
        absolute_layout = QHBoxLayout()
        absolute_layout.setContentsMargins(0, 0, 0, 0)
        absolute_layout.addWidget(QLabel("Density"))
        absolute_layout.addWidget(self.cme_density_spin)
        absolute_layout.addWidget(QLabel("Temperature"))
        absolute_layout.addWidget(self.cme_temperature_spin)
        self.absolute_row.setLayout(absolute_layout)

        add_form.addRow("HEEQ lon / lat", lon_lat_row)
        add_form.addRow("Speed / width", speed_width_row)
        add_form.addRow("Launch (day / datetime UTC)", launch_row)
        add_form.addRow("Thickness / initial height", size_row)
        add_form.addRow("CME duration", duration_row)
        add_form.addRow("Profile type", self.profile_type_combo)
        add_form.addRow("Active plasma mode", self.cme_plasma_mode_combo)
        add_form.addRow("Fraction values", self.fraction_row)
        add_form.addRow("Absolute values", self.absolute_row)
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

        layout.addWidget(load_box)
        layout.addWidget(add_box)
        layout.addWidget(list_box)
        self.setLayout(layout)
        self._sync_launch_datetime_from_day()
        self._on_cme_plasma_mode_changed(self.cme_plasma_mode_combo.currentText())
        self._update_initial_height_style()

    def load_cone_file(self):
        """Load CMEs from a cone2bc .in file and append them to the CME list."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select cone file",
            str(EXAMPLE_INPUTS_DIR),
            "Cone input (*.in)",
        )

        if not filepath:
            return

        self._apply_cone_file(filepath)

    def _apply_cone_file(
        self,
        filepath: str,
        replace_existing_loaded: bool = True,
        set_status: bool = True,
    ):
        """Apply a selected cone file path and load/refresh cone-file CMEs."""
        self.cone_file_edit.setText(filepath)

        try:
            cme_params = sin.import_cone2bc_parameters(filepath)
            if not cme_params:
                if set_status:
                    self.load_cone_status_label.setText("No CMEs found in selected cone file.")
                return

            if replace_existing_loaded:
                self._cmes = [cme for cme in self._cmes if cme.get("source") != "cone_file"]

            loaded = 0
            for cme in cme_params.values():
                launch_time = Time(cme["ldates"]).to_datetime()
                if launch_time.tzinfo is not None:
                    launch_time = launch_time.replace(tzinfo=None)

                delta_days = (launch_time - self.model_start_datetime).total_seconds() / 86400.0

                self._cmes.append(
                    {
                        "longitude": float(cme.get("lon", 0.0)),
                        "latitude": float(cme.get("lat", 0.0)),
                        "speed": float(cme.get("vcld", 800.0)),
                        "width": float(2.0 * cme.get("rmajor", 30.0)),
                        "t_launch_day": delta_days,
                        "t_launch_datetime": launch_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "thickness_rs": 0.0,
                        "initial_height_rs": 21.5,
                        "cme_expansion": False,
                        "cme_fixed_duration": True,
                        "fixed_duration_hr": 12.0,
                        "profile_type": "square",
                        "plasma_mode": "Fraction of ambient",
                        "density_fraction": 1.0,
                        "temperature_fraction": 1.0,
                        "cme_density_pcc": np.nan,
                        "cme_temperature_k": np.nan,
                        "source": "cone_file",
                    }
                )
                loaded += 1

            if set_status:
                self.load_cone_status_label.setText(f"Loaded {loaded} CME(s) from cone file.")
            self._refresh_cme_list()
        except Exception as exc:
            if set_status:
                self.load_cone_status_label.setText("Failed to load cone file. See terminal output.")
                self.load_cone_status_label.setToolTip(str(exc))

    def set_model_start_datetime(self, model_start: object):
        """Update model start reference and refresh launch datetime from day offset."""
        if isinstance(model_start, datetime.datetime):
            new_start = model_start.replace(tzinfo=None)
            if self._last_model_start_datetime == new_start:
                return

            self.model_start_datetime = new_start
            self._last_model_start_datetime = new_start
            self._sync_launch_datetime_from_day()
            self._reload_cone_file_for_updated_start()

    def _reload_cone_file_for_updated_start(self):
        """Re-load cone-file CMEs so their launch offsets follow the current model start."""
        filepath = self.cone_file_edit.text().strip()
        if not filepath:
            return

        self._apply_cone_file(filepath, replace_existing_loaded=True, set_status=False)
        self.load_cone_status_label.setText("Cone file reloaded for updated model start time.")

    def set_model_inner_boundary(self, rmin_rs: float):
        """Update reference inner boundary and refresh CME initial-height highlighting."""
        self.model_inner_boundary_rs = float(rmin_rs)
        self._update_initial_height_style()

    def set_model_run_duration_days(self, run_days: float):
        """Update run duration used to flag CMEs outside the simulation window."""
        self.model_run_duration_days = float(run_days)
        self._refresh_cme_list()

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

    def _on_cme_plasma_mode_changed(self, mode_text: str):
        """Keep all fields selectable, but visually mark which plasma mode is active."""
        use_absolute = mode_text == "Absolute values"
        active_style = "QWidget { background-color: rgba(20, 120, 20, 28); border-radius: 4px; }"
        inactive_style = "QWidget { background-color: rgba(120, 120, 120, 14); border-radius: 4px; }"
        self.fraction_row.setStyleSheet(inactive_style if use_absolute else active_style)
        self.absolute_row.setStyleSheet(active_style if use_absolute else inactive_style)

    def _update_initial_height_style(self):
        """Highlight CME initial height in red when it differs from model inner boundary."""
        differs = abs(self.cme_initial_height_spin.value() - self.model_inner_boundary_rs) > 1.0e-6
        if differs:
            self.cme_initial_height_spin.setStyleSheet(
                "QDoubleSpinBox { color: #b22222; font-weight: 600; }"
            )
            self.cme_initial_height_spin.setToolTip(
                "CME initial height differs from model inner boundary."
            )
        else:
            self.cme_initial_height_spin.setStyleSheet("")
            self.cme_initial_height_spin.setToolTip("")

    def _refresh_cme_list(self):
        """Refresh visible CME list from internal state."""
        self.cme_list_widget.clear()
        for idx, cme in enumerate(self._cmes, start=1):
            line = (
                f"{idx:02d}: t={cme['t_launch_day']} day, lon={cme['longitude']} deg, "
                f"lat={cme['latitude']} deg, v={cme['speed']} km/s, width={cme['width']} deg, "
                f"dt={cme['t_launch_datetime']} UTC"
            )
            item = QListWidgetItem(line)
            t_launch_day = float(cme.get("t_launch_day", 0.0))
            if t_launch_day < 0.0 or t_launch_day > self.model_run_duration_days:
                item.setForeground(QColor("#b22222"))
            self.cme_list_widget.addItem(item)

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
                "plasma_mode": self.cme_plasma_mode_combo.currentText(),
                "density_fraction": self.density_fraction_spin.value(),
                "temperature_fraction": self.temperature_fraction_spin.value(),
                "cme_density_pcc": self.cme_density_spin.value(),
                "cme_temperature_k": self.cme_temperature_spin.value(),
                "source": "manual",
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
        self.ambient_tab = AmbientSolarWindTab()
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
        self.model_tab.start_datetime.dateTimeChanged.connect(
            self._on_model_start_datetime_input_changed
        )
        self.model_tab.include_bpol_toggle.toggled.connect(self.ambient_tab.set_include_bpol)
        self.model_tab.start_datetime_updated.connect(self.cme_tab.set_model_start_datetime)
        self.model_tab.rmin_spin.valueChanged.connect(self.ambient_tab.set_model_inner_boundary)
        self.model_tab.latitude_spin.valueChanged.connect(self.ambient_tab.set_model_latitude)
        self.model_tab.rmin_spin.valueChanged.connect(self.cme_tab.set_model_inner_boundary)
        self.model_tab.simtime_spin.valueChanged.connect(self.cme_tab.set_model_run_duration_days)
        self._on_1d_mode_changed(self.model_tab.one_d_toggle.isChecked())
        self.ambient_tab.set_model_inner_boundary(self.model_tab.rmin_spin.value())
        self.ambient_tab.set_model_latitude(self.model_tab.latitude_spin.value())
        self.ambient_tab.set_include_bpol(self.model_tab.include_bpol_toggle.isChecked())
        self.cme_tab.set_model_start_datetime(self.model_tab.start_datetime.dateTime().toPyDateTime())
        self.cme_tab.set_model_inner_boundary(self.model_tab.rmin_spin.value())
        self.cme_tab.set_model_run_duration_days(self.model_tab.simtime_spin.value())
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

        self.ambient_tab.status_message.connect(self.status_label.setText)
        self.ambient_tab.error_message.connect(self._on_ambient_error)
        self.ambient_tab.start_time_selected.connect(self.model_tab.set_start_datetime)
        self.ambient_tab.emit_active_map_time_if_enabled()

    def _on_model_start_datetime_input_changed(self, _qdt: QDateTime):
        """Ensure manual datetime edits immediately refresh cone-file CME launch offsets."""
        self.cme_tab.set_model_start_datetime(self.model_tab.start_datetime.dateTime().toPyDateTime())

    def _build_generated_code(self):
        """Create a runnable Python script from current GUI state."""
        state = self.model_tab.get_state()
        ambient = self.ambient_tab.get_state()
        cmes = self.cme_tab.get_cmes()
        lines = [
            "import datetime",
            "import numpy as np",
            "import astropy.units as u",
            "import astropy.constants as const",
            "import surf.surf as s",
            "",
            "# Generated by SURF GUI",
            f"rmin = {state['rmin']} * u.solRad",
            f"rmax = {state['rmax']} * u.solRad",
            f"lon_start = {state['lon_min']} * u.deg",
            f"lon_stop = {state['lon_max']} * u.deg",
            f"latitude = {state['latitude']} * u.deg",
            f"frame = {state['frame']!r}",
            f"simtime = {state['simtime_days']} * u.day",
            f"cr_num = {state['cr_num']}",
            f"cr_lon_init = {state['cr_lon_init_deg']} * u.deg",
            f"start_time = datetime.datetime.fromisoformat({state['start_datetime']!r})",
            "",
            "# Ambient solar wind boundary.",
        ]

        if ambient["source"] == "user_specified":
            boundary_lines = [
                f"v_boundary = np.array({ambient['speed_profile_kms']!r}) * (u.km / u.s)",
                "",
            ]
        elif ambient["source"] == "mas":
            include_bpol = state.get("include_bpol", False)
            map_to_rmin = ambient.get("decelerate_to_inner_boundary", True)
            boundary_lines = ["import surf.surf_inputs as sin"]
            if include_bpol:
                boundary_lines.extend(
                    [
                        f"v_boundary = sin.get_MAS_long_profile({ambient['cr_num']}, latitude)",
                        f"b_boundary = sin.get_MAS_br_long_profile({ambient['cr_num']}, latitude)",
                        "if len(b_boundary) != len(v_boundary):",
                        "    b_lon = np.linspace(0.0, 360.0, len(b_boundary), endpoint=False)",
                        "    v_lon = np.linspace(0.0, 360.0, len(v_boundary), endpoint=False)",
                        "    b_boundary = np.interp(v_lon, b_lon, np.asarray(b_boundary), period=360.0)",
                    ]
                )
                if map_to_rmin:
                    boundary_lines.append(
                        "v_boundary, b_boundary = sin.map_v_boundary_inwards("
                        "v_boundary, 30.0 * u.solRad, rmin, b_orig=b_boundary)"
                    )
                else:
                    boundary_lines.append(
                        "# Using MAS speed and bpol at 30 Rs directly (no mapping to rmin)."
                    )
            else:
                boundary_lines.append(f"v_boundary = sin.get_MAS_long_profile({ambient['cr_num']}, latitude)")
                boundary_lines.append(
                    "v_boundary = sin.map_v_boundary_inwards(v_boundary, 30.0 * u.solRad, rmin)"
                    if map_to_rmin
                    else "# Using MAS speed at 30 Rs directly (no mapping to rmin)."
                )
            boundary_lines.append("")
        elif ambient["source"] == "wsa":
            include_bpol = state.get("include_bpol", False)
            map_to_rmin = ambient.get("decelerate_to_inner_boundary", True)
            boundary_lines = ["import surf.surf_inputs as sin"]
            if include_bpol:
                boundary_lines.extend(
                    [
                        f"v_boundary = sin.get_WSA_long_profile(r{ambient['filepath']!r}, latitude)",
                        f"b_boundary = sin.get_WSA_br_long_profile(r{ambient['filepath']!r}, latitude)",
                        "if len(b_boundary) != len(v_boundary):",
                        "    b_lon = np.linspace(0.0, 360.0, len(b_boundary), endpoint=False)",
                        "    v_lon = np.linspace(0.0, 360.0, len(v_boundary), endpoint=False)",
                        "    b_boundary = np.interp(v_lon, b_lon, np.asarray(b_boundary), period=360.0)",
                    ]
                )
                if map_to_rmin:
                    boundary_lines.append(
                        "v_boundary, b_boundary = sin.map_v_boundary_inwards("
                        "v_boundary, 21.5 * u.solRad, rmin, b_orig=b_boundary)"
                    )
                else:
                    boundary_lines.append(
                        "# Using WSA speed and bpol at 21.5 Rs directly (no mapping to rmin)."
                    )
            else:
                boundary_lines.append(
                    f"v_boundary = sin.get_WSA_long_profile(r{ambient['filepath']!r}, latitude)"
                )
                boundary_lines.append(
                    "v_boundary = sin.map_v_boundary_inwards(v_boundary, 21.5 * u.solRad, rmin)"
                    if map_to_rmin
                    else "# Using WSA speed at 21.5 Rs directly (no mapping to rmin)."
                )
            boundary_lines.append("")
        elif ambient["source"] == "insitu_backmapped":
            boundary_lines = [
                "# Boundary is initialized from OMNI observations using Example 27 helpers.",
                "insitu_rmin = rmin",
                "",
            ]
        elif ambient["source"] == "omni":
            boundary_lines = [
                "import surf.surf_insitu as sinsit",
                "import surf.surf_inputs as sin",
                "omni_rmin = 215.0 * u.solRad"
                if ambient.get("use_215_inner_boundary", True)
                else "omni_rmin = rmin",
                "omni_end_time = start_time + datetime.timedelta(days=simtime.to_value(u.day))",
                "omni_time_grid, omni_vcarr, omni_bcarr = sinsit.generate_vCarr_from_OMNI(",
                "    start_time, omni_end_time",
                ")",
                "",
            ]
        elif ambient["source"] == "cortom":
            boundary_lines = [
                "import surf.surf_inputs as sin",
                (
                    f"v_boundary = sin.get_CorTom_long_profile(r{ambient['filepath']!r}, "
                    "latitude)"
                ),
                (
                    "v_boundary = sin.map_v_boundary_inwards(v_boundary, 8.0 * u.solRad, rmin)"
                    if ambient.get("decelerate_to_inner_boundary", True)
                    else "# Using CorTom speed at 8 Rs directly (no mapping to rmin)."
                ),
                "",
            ]
        else:
            boundary_lines = [
                "# Selected ambient source is not implemented yet in the GUI;",
                "# using the default uniform profile until that source is wired.",
                "v_boundary = np.ones(128) * 400 * (u.km / u.s)",
                "",
            ]

        lines.extend(boundary_lines)

        if ambient["source"] == "insitu_backmapped":
            lines.insert(2, "import surf.surf_insitu as sinsit")
            if ambient["mode"] == "reconstruction":
                lines.extend(
                    [
                        "model = sinsit.omniSURF_reconstruction(",
                        "    start_time,",
                        "    start_time + datetime.timedelta(days=simtime.to_value(u.day)),",
                        "    rmin=insitu_rmin,",
                        "    rmax=rmax,",
                        "    dt_scale=4,",
                        f"    run_2d={str(not state['is_1d'])},",
                        ")",
                    ]
                )
            else:
                lines.extend(
                    [
                        "model = sinsit.omniSURF_forecast(",
                        "    start_time,",
                        "    simtime=simtime,",
                        "    rmin=insitu_rmin,",
                        "    rmax=rmax,",
                        "    dt_scale=4,",
                        f"    run_2d={str(not state['is_1d'])},",
                        ")",
                    ]
                )
        elif ambient["source"] == "omni":
            include_bpol = state.get("include_bpol", False)
            if state["is_1d"]:
                omni_call_lines = [
                    "model = sin.set_time_dependent_boundary(",
                    "    omni_vcarr,",
                    "    omni_time_grid,",
                    "    start_time,",
                    "    simtime,",
                    "    r_min=omni_rmin,",
                    "    r_max=rmax,",
                    "    dt_scale=4,",
                    "    latitude=latitude,",
                    "    lon_out=0.0 * u.deg,",
                ]
                if include_bpol:
                    omni_call_lines.append("    bgrid_Carr=omni_bcarr,")
                omni_call_lines.extend(
                    [
                        "    track_cmes=True,",
                        ")",
                    ]
                )
                lines.extend(omni_call_lines)
            else:
                omni_call_lines = [
                    "model = sin.set_time_dependent_boundary(",
                    "    omni_vcarr,",
                    "    omni_time_grid,",
                    "    start_time,",
                    "    simtime,",
                    "    r_min=omni_rmin,",
                    "    r_max=rmax,",
                    "    dt_scale=4,",
                    "    latitude=latitude,",
                    "    frame=frame,",
                    "    lon_start=lon_start,",
                    "    lon_stop=lon_stop,",
                ]
                if include_bpol:
                    omni_call_lines.append("    bgrid_Carr=omni_bcarr,")
                omni_call_lines.extend(
                    [
                        "    track_cmes=True,",
                        ")",
                    ]
                )
                lines.extend(omni_call_lines)
        else:
            lines.extend(
                [
                    "model = s.SURF(",
                    "    v_boundary=v_boundary,",
                    *(["    b_boundary=b_boundary,"] if state.get("include_bpol", False) else []),
                    "    cr_num=cr_num,",
                    "    cr_lon_init=cr_lon_init,",
                    "    r_min=rmin,",
                    "    r_max=rmax,",
                ]
            )

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
                    "    simtime=simtime,",
                    "    dt_scale=4,",
                    ")",
                ]
            )

        lines.extend(["", "cme_list = []"])

        for idx, cme in enumerate(cmes):
            use_absolute_plasma = cme.get("plasma_mode") == "Absolute values"
            plasma_args = []
            if use_absolute_plasma:
                plasma_args.extend(
                    [
                        (
                            "cme_density=("
                            f"{cme.get('cme_density_pcc', np.nan)}*(1/(u.cm**3))*const.m_p"
                            ").to(u.kg/(u.m**3))"
                        ),
                        f"cme_temperature={cme.get('cme_temperature_k', np.nan)}*u.K",
                    ]
                )
            else:
                plasma_args.extend(
                    [
                        f"density_fraction={cme.get('density_fraction', 1.0)}",
                        f"temperature_fraction={cme.get('temperature_fraction', 1.0)}",
                    ]
                )

            cme_arg_parts = [
                f"t_launch={cme['t_launch_day']}*u.day",
                f"longitude={cme['longitude']}*u.deg",
                f"latitude={cme['latitude']}*u.deg",
                f"width={cme['width']}*u.deg",
                f"v={cme['speed']}*(u.km/u.s)",
                f"thickness={cme['thickness_rs']}*u.solRad",
                f"initial_height={cme['initial_height_rs']}*u.solRad",
                f"cme_expansion={cme['cme_expansion']}",
                f"cme_fixed_duration={cme['cme_fixed_duration']}",
                f"fixed_duration={cme['fixed_duration_hr']}*u.hour",
            ]
            cme_arg_parts.extend(plasma_args)
            cme_arg_parts.append(f"profile_type={cme['profile_type']!r}")

            lines.extend(
                [
                    f"cme_{idx} = s.ConeCME({', '.join(cme_arg_parts)})",
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
        self._sync_model_inner_boundary_for_omni()
        code_text = self._build_generated_code()
        dialog = CodeDialog(code_text, self)
        dialog.exec()

    def show_terminal_output(self):
        """Show captured output from the most recent SURF run in a separate window."""
        dialog = TerminalOutputDialog(self.last_terminal_output, self)
        dialog.exec()

    def run_surf(self):
        """Execute generated SURF code and update UI state accordingly."""
        self._sync_model_inner_boundary_for_omni()
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

    def _sync_model_inner_boundary_for_omni(self):
        """Force model inner boundary to 215 Rs when OMNI source is configured to use it."""
        ambient = self.ambient_tab.get_state()
        if ambient.get("source") != "omni":
            return
        if not ambient.get("use_215_inner_boundary", True):
            return

        if abs(self.model_tab.rmin_spin.value() - 215.0) > 1.0e-6:
            self.model_tab.rmin_spin.setValue(215.0)

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

    def _on_ambient_error(self, error_text: str):
        """Capture ambient-tab tracebacks and surface a concise status message."""
        self._append_terminal_output(error_text)
        self.status_label.setText(
            "Ambient solar wind action failed. Open 'Show Terminal Output' for details."
        )

    def plot_map(self):
        """Generate a 2D map plot matching notebook sa.plot usage."""
        if self.last_model is None:
            self.status_label.setText("Run SURF first to enable plotting.")
            return

        try:
            plot_time = self.visualisation_tab.map_time_spin.value() * u.day
            plot_rmax = (
                self.visualisation_tab.map_rmax_spin.value()
                if self.visualisation_tab.map_limit_rmax_toggle.isChecked()
                else None
            )
            sa.plot(
                self.last_model,
                plot_time,
                minimalplot=self.visualisation_tab.map_minimalplot_toggle.isChecked(),
                plotHCS=self.visualisation_tab.map_plot_hcs_toggle.isChecked(),
                annotateplot=self.visualisation_tab.map_annotate_toggle.isChecked(),
                trace_earth_connection=self.visualisation_tab.map_trace_earth_toggle.isChecked(),
                plot_rmax=plot_rmax,
            )
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
