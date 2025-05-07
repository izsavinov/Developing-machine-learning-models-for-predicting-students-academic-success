// Ваши переменные с цветами
var startColor = "#00FFF0";
var endColor = "#000AFF";
var fontcolor = "#000000";
var linecolor = "#DEDEDE";

function initCustomSelect(selectElement) {
  const customSelect = document.createElement('div');
  customSelect.className = 'custom-select course-select'; // Добавил специальный класс

  const selectedOption = document.createElement('div');
  selectedOption.className = 'selected-option';

  const optionsContainer = document.createElement('div');
  optionsContainer.className = 'options-container';

  // Такая же SVG стрелка как в параметрах
  const arrowSvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#666" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M6 9l6 6 6-6"/>
  </svg>`;

  // Переносим опции
  Array.from(selectElement.options).forEach((opt, index) => {
    const optionElement = document.createElement('div');
    optionElement.className = `option ${index === 0 ? 'selected' : ''}`;
    optionElement.textContent = opt.text;
    optionElement.dataset.value = opt.value;
    optionsContainer.appendChild(optionElement);
  });

  // Собираем структуру
  selectedOption.innerHTML = `
    <span>${selectElement.selectedOptions[0]?.text || 'Выберите курс'}</span>
    ${arrowSvg}
  `;

  customSelect.appendChild(selectedOption);
  customSelect.appendChild(optionsContainer);
  selectElement.replaceWith(customSelect);

  // Логика работы
  selectedOption.addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.options-container').forEach(container => {
      if (container !== optionsContainer) container.style.display = 'none';
    });
    optionsContainer.style.display = optionsContainer.style.display === 'block' ? 'none' : 'block';

    // Анимация стрелки
    const arrow = selectedOption.querySelector('svg');
    if (optionsContainer.style.display === 'block') {
      arrow.style.transform = 'rotate(180deg)';
    } else {
      arrow.style.transform = 'rotate(0)';
    }
  });

  optionsContainer.addEventListener('click', (e) => {
    if (e.target.classList.contains('option')) {
      const value = e.target.dataset.value;
      const text = e.target.textContent;

      // Обновляем отображение
      selectedOption.querySelector('span').textContent = text;
      optionsContainer.style.display = 'none';
      selectedOption.querySelector('svg').style.transform = 'rotate(0)';

      // Обновляем оригинальный select
      selectElement.value = value;

      // Триггерим событие change
      const event = new Event('change', { bubbles: true });
      selectElement.dispatchEvent(event);
    }
  });

  // Закрытие при клике вне селекта
  document.addEventListener('click', (e) => {
    if (!customSelect.contains(e.target)) {
      optionsContainer.style.display = 'none';
      selectedOption.querySelector('svg').style.transform = 'rotate(0)';
    }
  });
}

function initParameterSelect(selectElement) {
  const customSelect = document.createElement('div');
  customSelect.className = 'custom-select parameter-select'; // Добавил специальный класс

  const selectedOption = document.createElement('div');
  selectedOption.className = 'selected-option';

  const optionsContainer = document.createElement('div');
  optionsContainer.className = 'options-container';

  // SVG стрелка (можно сделать другой стиль)
  const arrowSvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#666" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M6 9l6 6 6-6"/>
  </svg>`;

  // Переносим опции
  Array.from(selectElement.options).forEach((opt, index) => {
    const optionElement = document.createElement('div');
    optionElement.className = `option ${index === 0 ? 'selected' : ''}`;
    optionElement.textContent = opt.text;
    optionElement.dataset.value = opt.value;
    optionsContainer.appendChild(optionElement);
  });

  // Собираем структуру
  selectedOption.innerHTML = `
    <span>${selectElement.selectedOptions[0]?.text || 'Выберите параметр'}</span>
    ${arrowSvg}
  `;

  customSelect.appendChild(selectedOption);
  customSelect.appendChild(optionsContainer);
  selectElement.replaceWith(customSelect);

  // Логика работы
  selectedOption.addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.options-container').forEach(container => {
      if (container !== optionsContainer) container.style.display = 'none';
    });
    optionsContainer.style.display = optionsContainer.style.display === 'block' ? 'none' : 'block';

    // Анимация стрелки
    const arrow = selectedOption.querySelector('svg');
    if (optionsContainer.style.display === 'block') {
      arrow.style.transform = 'rotate(180deg)';
    } else {
      arrow.style.transform = 'rotate(0)';
    }
  });

  optionsContainer.addEventListener('click', (e) => {
    if (e.target.classList.contains('option')) {
      const value = e.target.dataset.value;
      const text = e.target.textContent;

      // Обновляем отображение
      selectedOption.querySelector('span').textContent = text;
      optionsContainer.style.display = 'none';
      selectedOption.querySelector('svg').style.transform = 'rotate(0)';

      // Обновляем оригинальный select
      selectElement.value = value;

      // Триггерим событие change
      const event = new Event('change', { bubbles: true });
      selectElement.dispatchEvent(event);

      // Перерисовываем график
      if (selectElement.id === 'parameter-select') {
        renderChart(value);
      }
    }
  });

  // Закрытие при клике вне селекта
  document.addEventListener('click', (e) => {
    if (!customSelect.contains(e.target)) {
      optionsContainer.style.display = 'none';
      selectedOption.querySelector('svg').style.transform = 'rotate(0)';
    }
  });
}
// Chart options configurations
var options_pie = {
  legend: {
    display: true,
    position: 'bottom',
    shape: "circle",
    align: 'center',
    labels: {
      fontColor: fontcolor
    }
  },
  title: {
    display: false,
    text: 'Bar Chart'
  }
};

var options_doughnut = {
  legend: {
    display: true,
    position: 'bottom',
    shape: "circle",
    align: 'center',
    labels: {
      fontColor: fontcolor
    }
  },
  title: {
    display: false,
    text: 'Bar Chart'
  },
  fontColor: fontcolor
};

var options_bar = {
  legend: {
    display: false
  },
  indexAxis: 'x',
  title: {
    display: false,
    text: 'Bar Chart'
  },
  scales: {
    yAxes: [{
      ticks: {
        beginAtZero: true,
        maxSteps: 10,
        fontColor: fontcolor,
        format: {
          style: 'percent'
        }
      },
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor
      }
    }],
    xAxes: [{
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor,
        display: false
      },
      ticks: {
        fontColor: fontcolor,
        format: {
          style: 'percent'
        }
      }
    }]
  }
};

var options_horizontalBar = {
  legend: {
    display: false,
    labels: {
      fontColor: fontcolor
    }
  },
  title: {
    display: false
  },
  scales: {
    xAxes: [{
      ticks: {
        beginAtZero: true,
        fontColor: fontcolor
      },
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor
      }
    }],
    yAxes: [{
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor,
        display: false
      },
      ticks: {
        beginAtZero: true,
        fontColor: fontcolor
      }
    }]
  }
};

var options_line = {
  legend: {
    display: true,
    position: 'bottom',
    labels: {
      fontColor: fontcolor
    }
  },
  indexAxis: 'x',
  title: {
    display: false,
    text: 'Bar Chart'
  },
  fontColor: fontcolor,
  gridLines: {
    color: linecolor
  },
  scales: {
    yAxes: [{
      ticks: {
        beginAtZero: true,
        fontColor: fontcolor
      },
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor
      }
    }],
    xAxes: [{
      gridLines: {
        color: linecolor,
        zeroLineColor: linecolor
      },
      ticks: {
        fontColor: fontcolor
      }
    }]
  }
};

function create_options_bubble(style) {
  return {
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        fontColor: fontcolor
      }
    },
    aspectRatio: 2.5,
    plugins: {
      title: {
        display: false,
        font: { size: 16, weight: 'bold' }
      }
    },
    title: {
      display: false
    },
    scales: {
      yAxes: [{
        gridLines: {
          color: linecolor,
          zeroLineColor: linecolor
        },
        ticks: {
          fontColor: fontcolor
        }
      }],
      xAxes: [{
        gridLines: {
          color: linecolor,
          zeroLineColor: linecolor
        },
        ticks: {
          fontColor: fontcolor,
          callback: function(value, index, values) {
            if (style == 'time') {
              var date = new Date(value);
              return date.toLocaleTimeString('it-IT');
            }
            if (style == 'date') {
              var date = new Date(value);
              return date.toLocaleDateString('en-GB');
            }
            if (style == 'datetime') {
              var date = new Date(value);
              return date.toLocaleString('en-GB');
            }
            return value;
          }
        }
      }]
    }
  };
}

// Color utility functions
function hex(c) {
  var s = "0123456789abcdef";
  var i = parseInt(c);
  if (i == 0 || isNaN(c)) return "00";
  i = Math.round(Math.min(Math.max(0, i), 255));
  return s.charAt((i - i % 16) / 16) + s.charAt(i % 16);
}

function convertToHex(rgb) {
  return hex(rgb[0]) + hex(rgb[1]) + hex(rgb[2]);
}

function trim(s) {
  return (s.charAt(0) == '#') ? s.substring(1, 7) : s;
}

function convertToRGB(hex) {
  var color = [];
  color[0] = parseInt((trim(hex)).substring(0, 2), 16);
  color[1] = parseInt((trim(hex)).substring(2, 4), 16);
  color[2] = parseInt((trim(hex)).substring(4, 6), 16);
  return color;
}

function generateColor(colorStart, colorEnd, colorCount) {
  var start = convertToRGB(colorStart);
  var end = convertToRGB(colorEnd);
  var len = colorCount;
  var alpha = 0.0;
  var saida = [];
  saida.push(colorEnd);

  for (var i = 0; i < len - 2; i++) {
    var c = [];
    alpha += (1.0 / len);
    c[0] = start[0] * alpha + (1 - alpha) * end[0];
    c[1] = start[1] * alpha + (1 - alpha) * end[1];
    c[2] = start[2] * alpha + (1 - alpha) * end[2];
    saida.push('#' + convertToHex(c).toString());
  }

  saida.push(colorStart);
  return saida;
}

var colors100 = generateColor(endColor, startColor, 100);
var options_array = [
  { id: 'pie', options: options_pie },
  { id: 'doughnut', options: options_doughnut },
  { id: 'bar', options: options_bar },
  { id: 'horizontalBar', options: options_horizontalBar },
  { id: 'line', options: options_line }
];

class Diagram {
  constructor(id, type, headers, values, values_x = [], x_style = '') {
    this.id = id;
    this.type = type;
    this.headers = headers;
    this.values = values;
    this.values_x = values_x;
    this.x_style = x_style;
    this.analyse_data_line();
  }

  getId() { return this.id; }
  getType() { return this.type; }
  getHeaders() { return this.headers; }
  getValues() { return this.values; }
  getValues_x() { return this.values_x; }

  analyse_data_line() {
    if (this.type == 'line' || this.type == 'area') {
      var values_x = this.values_x;
      var values = [];

      for (var i = 0; i < this.values.length; i++) {
        var values_t = [];

        for (var j = 0; j < values_x.length; j++) {
          var value = 'null';
          for (var k = 0; k < this.values[i].length; k++) {
            if (values_x[j] == this.values[i][k][0]) {
              value = this.values[i][k][1];
              break;
            }
          }
          values_t.push(value);
        }

        for (var k = 0; k < values_t.length; k++) {
          if (values_t[k] == 'null') {
            if (k - 1 > -1) {
              var flag = 0;
              for (var t = k + 1; t < values_t.length; t++) {
                if (values_t[t] != 'null') {
                  flag = 1;
                  break;
                }
              }
              if (flag == 1 && values_t[k - 1] != 'null') {
                var r = (values_t[t] - values_t[k - 1]) / (t - (k - 1));
                values_t[k] = values_t[k - 1] + r;
              }
            }
          }
        }
        values.push(values_t);
      }
      this.values = values;
    }
  }

  create_datasets(type) {
    var backgroundColors = generateColor(endColor, startColor, this.headers.length);
    var fill = this.type == 'area';
    var datasets = [];
    var labels = this.headers;

    if (type == 'pie' || type == 'doughnut' || type == 'bar' || type == 'horizontalBar') {
      datasets = [{
        label: { display: true },
        data: this.values,
        backgroundColor: backgroundColors,
        borderWidth: 0,
        borderColor: backgroundColors
      }];
    }
    else if (type == 'line') {
      var radius = fill ? 0 : 3;
      for (let i = 0; i < this.values.length; i++) {
        datasets.push({
          label: this.headers[i],
          backgroundColor: backgroundColors[i],
          data: this.values[i],
          pointBackgroundColor: backgroundColors[i],
          borderColor: backgroundColors[i],
          fill: fill,
          borderWidth: 0,
          pointRadius: radius
        });
      }
    }
    else if (type == 'bubble') {
      for (let i = 0; i < this.values.length; i++) {
        var data = [];
        for (let j = 0; j < this.values[i].length; j++) {
          data.push({
            x: this.values[i][j][0],
            y: this.values[i][j][1],
            r: this.values[i][j][2]
          });
        }
        datasets.push({
          label: this.headers[i],
          backgroundColor: backgroundColors[i],
          data: data,
          borderColor: backgroundColors[i],
          borderWidth: 0
        });
      }
    }
    return datasets;
  }

  draw() {
    var element = document.getElementById(this.id);
    var type = this.type == 'area' ? 'line' : this.type;
    var labels = type == 'line' ? this.values_x : this.headers;
    var datasets = this.create_datasets(type);

    var options = this.type == 'bubble'
      ? create_options_bubble(this.x_style)
      : options_array.find(obj => obj.id === type).options;

    this.chart = new Chart(element, {
      type: type,
      data: {
        labels: labels,
        datasets: datasets
      },
      options: options
    });
  }

  update(values, values_x) {
    this.values = values;
    this.analyse_data_line();
    this.values_x = values_x;
    this.chart.data.datasets = this.create_datasets(this.type == 'area' ? 'line' : this.type);
    this.chart.update();
  }

  add(header, value) {
    this.headers.push(header);
    this.values.push(value);
    this.analyse_data_line();
    this.chart.destroy();
    this.draw();
  }

  pop(index) {
    this.headers.splice(index - 1, 1);
    this.values.splice(index - 1, 1);
    this.chart.destroy();
    this.draw();
  }

  remove() {
    var element = document.getElementById(this.id);
    element.remove();
  }
}

// Chrome extension setup
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((error) => console.error(error));

chrome.tabs.onUpdated.addListener(async (tabId, info, tab) => {
  if (!tab.url) return;
  const url = new URL(tab.url);
  const GOOGLE_ORIGIN = 'https://www.google.com';

  if (url.origin === GOOGLE_ORIGIN) {
    await chrome.sidePanel.setOptions({
      tabId,
      path: 'sidepanel.html',
      enabled: true
    });
  } else {
    await chrome.sidePanel.setOptions({
      tabId,
      enabled: false
    });
  }
});

// Main application
document.addEventListener("DOMContentLoaded", () => {
  var charts = [];
  const menuBtn = document.getElementById('menuBtn');
  const commandPanel = document.getElementById('commandPanel');
  const exitBtn = document.getElementById('exitBtn');
  const exitModal = document.getElementById('exitModal');
  const cancelExit = document.getElementById('cancelExit');
  const confirmExit = document.getElementById('confirmExit');

  // UI event handlers
  menuBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    commandPanel.classList.toggle('hidden');
  });

  document.addEventListener('click', (e) => {
    if (!commandPanel.contains(e.target) && e.target !== menuBtn) {
      commandPanel.classList.add('hidden');
    }
  });

  exitBtn.addEventListener('click', () => {
    commandPanel.classList.add('hidden');
    exitModal.classList.remove('hidden');
  });

  cancelExit.addEventListener('click', () => {
    exitModal.classList.add('hidden');
  });

  confirmExit.addEventListener('click', () => {
    exitModal.classList.add('hidden');
    console.log('Пользователь вышел');
  });

  // Auth and chart related elements
  const authForm = document.getElementById("auth-form");
  const dashboard = document.getElementById("dashboard");
  const loginForm = document.getElementById("login-form");
  const logoutBtn = document.getElementById("confirmExit");
  const courseSelect = document.getElementById("course-select");
  const parameterSelect = document.getElementById("parameter-select");
  const chartCanvas = document.getElementById("chart");
  const currentAssessment = document.getElementById("current-assessment");
  const predictiveAssessment = document.getElementById("predictive-assessment");
  const currentAssessmentMean = document.getElementById("current-assessment-mean");
  const predictiveAssessmentMean = document.getElementById("predictive-assessment-mean");
  let error_fio = document.getElementById("error_fio");
  let error_password = document.getElementById("error_password");
  let input_fio = document.getElementById("fio");
  let input_password = document.getElementById("password");

  let currentChart = null;

  // Utility functions
  function encodeBase64(str) {
    return btoa(
      encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, (match, p1) =>
        String.fromCharCode(parseInt(p1, 16)))
    );
  }

  function checkAuth() {
    chrome.cookies.get({ url: "http://212.41.9.83", name: "fio" }, (cookie) => {
      if (cookie) {
        showDashboard(cookie.value);
      } else {
        showLoginForm();
      }
    });
  }

  function showLoginForm() {
    authForm.classList.remove("hidden");
    dashboard.classList.add("hidden");
    var el_data = document.getElementById("data");
        el_data.className = "hidden";
        var el_no_data = document.getElementById("no_data");
        el_no_data.className = "hidden";
    input_fio.value = "";
    input_password.value = "";
  }

  function showDashboard(fio) {
    document.getElementById("user-fio").textContent = fio;
    authForm.classList.add("hidden");
    dashboard.classList.remove("hidden");
    loadAvailableCourses();
  }

  function form_validation(fio, password) {
    let valid = true;

    if (fio == "") {
      error_fio.className = "error";
      input_fio.className = "input_error";
      valid = false;
    }
    if (password == "") {
      error_password.className = "error";
      input_password.className = "input_error";
      valid = false;
    }

    return valid;
  }

  // Input event listeners
  document.getElementById('fio').addEventListener('input', function() {
    error_fio.className = "error hidden";
    input_fio.className = "input";
    var error_data = document.getElementById("error_data");
      error_data.className = "error unvisible";
  });

  document.getElementById('password').addEventListener('input', function() {
    error_password.className = "error hidden";
    input_password.className = "input";
    var error_data = document.getElementById("error_data");
      error_data.className = "error unvisible";
  });

  loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fio = document.getElementById("fio").value;
  const password = document.getElementById("password").value;

  if (!form_validation(fio, password)) return;

  const credentials = encodeBase64(`${fio}:${password}`);

  try {
    const response = await fetch("http://212.41.9.83/predict_grades/api/v1/login", {
      method: "POST",
      headers: { Authorization: `Basic ${credentials}` },
    });

    if (response.ok) {
      try {
        chrome.cookies.set({ url: "http://212.41.9.83", name: "fio", value: fio });
        chrome.cookies.set({ url: "http://212.41.9.83", name: "password", value: encodeBase64(password) });
        showDashboard(fio);
      } catch (cookieError) {
        console.error("Ошибка при установке кук:", cookieError);
      }
    } else {
      const errorText = await response.text(); // Получаем текст ошибки с сервера
      console.error("Ошибка при входе:", response.status, errorText);
      var error_data = document.getElementById("error_data");
      error_data.className = "error";
    }
  } catch (networkError) {
    console.error("Ошибка сети при выполнении запроса:", networkError);
  }
});

  // Logout handler
  logoutBtn.addEventListener("click", () => {
    chrome.cookies.remove({ url: "http://212.41.9.83", name: "fio" });
    chrome.cookies.remove({ url: "http://212.41.9.83", name: "password" });
    showLoginForm();
  });

  // Course data functions
  async function getCookie(name) {
    return new Promise((resolve) => {
      chrome.cookies.get({ url: "http://212.41.9.83", name: name }, (cookie) => {
        resolve(cookie ? cookie.value : null);
      });
    });
  }

  async function loadAvailableCourses() {
    const fio = await getCookie("fio");
    const password = decodeURIComponent(atob(await getCookie("password")));

    const credentials = encodeBase64(`${fio}:${password}`);
    const response = await fetch("http://212.41.9.83/predict_grades/api/v1/available_courses", {
      method: "GET",
      headers: { Authorization: `Basic ${credentials}` },
    });

   if (response.ok) {
    const data = await response.json();
    if (data.available_courses.length == 0) {

        var el_data = document.getElementById("data");
        el_data.className = "hidden";
        var el_no_data = document.getElementById("no_data");
        el_no_data.className = "";
    }
    else {
        var el_data = document.getElementById("data");
        el_data.className = "";
        var el_no_data = document.getElementById("no_data");
        el_no_data.className = "hidden";
    courseSelect.innerHTML = "";
    data.available_courses.forEach((course) => {
      const option = document.createElement("option");
      option.value = course.id;
      option.textContent = course.name;
      courseSelect.appendChild(option);
    });

    // Инициализируем кастомный селект только после заполнения options
    initCustomSelect(courseSelect);
    loadCourseData(courseSelect.value);}
  } else {
    alert("Ошибка загрузки курсов");
  }
}

  function setAssessmentColor(element, value) {
    if (value < 4.0) {
        element.style.color = '#db4848';
    } else if (value >= 4.0 && value <= 5.99) {
        element.style.color = '#fccb1d';
    } else if (value >= 6) {
        element.style.color = "#30bf57";
    }
}

  async function loadCourseData(courseId) {
    const fio = await getCookie("fio");
    const password = decodeURIComponent(atob(await getCookie("password")));

    const credentials = encodeBase64(`${fio}:${password}`);
    const response = await fetch("http://212.41.9.83/predict_grades/api/v1/course_data/" + courseId, {
      method: "GET",
      headers: {
        Authorization: `Basic ${credentials}`,
        "Content-Type": "application/json"
      },
    });

    if (response.ok) {
      const data = await response.json();
      currentAssessment.textContent = data.current_assessment.toFixed(2);
setAssessmentColor(currentAssessment, data.current_assessment);

predictiveAssessment.textContent = data.predictive_assessment.toFixed(2);
setAssessmentColor(predictiveAssessment, data.predictive_assessment);

currentAssessmentMean.textContent = data.current_assessment_mean.toFixed(2);
setAssessmentColor(currentAssessmentMean, data.current_assessment_mean);

predictiveAssessmentMean.textContent = data.predictive_assessment_mean.toFixed(2);
setAssessmentColor(predictiveAssessmentMean, data.predictive_assessment_mean);
      parameterSelect.innerHTML = "";
    data.charts.forEach((option) => {
      const optionElement = document.createElement("option");
      optionElement.value = option.option;
      optionElement.textContent = option.option;
      parameterSelect.appendChild(optionElement);
    });

    // Инициализируем кастомный селект для параметров
    initParameterSelect(parameterSelect);

      charts = data.charts;

      if (charts.length > 0) {
        renderChart(charts[0].option, 'student');
      }
    } else {
      alert("Ошибка загрузки данных курса");
    }
  }


let currentChartType = 'bar'; // По умолчанию столбчатый

// Обработчики кнопок переключения типа графика
document.querySelectorAll('.chart-type-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentChartType = btn.dataset.type;

    // Перерисовываем график с новым типом
    const selectedOption = parameterSelect.value;
    renderChart(selectedOption);
  });
});

function renderChart(selectedOption) {
  if (currentChart) {
    currentChart.destroy();
  }

  const chartData = charts.find(chart => chart.option === selectedOption);
  if (!chartData) {
    console.error("Данные для графика не найдены:", selectedOption);
    return;
  }

  // Получаем данные студента и группы
  const studentData = chartData.student || [];
  const groupData = chartData.group_average || [];

  const labels = studentData.map(item => item.period.toString());

  // Подготавливаем данные
  const studentValues = studentData.map(item => item.value);
  const groupValues = groupData.map(item => item.value);

  // Цвета для студента и группы
  const studentColor = '#86D3FF'; // Синий
  const groupColor = '#FCC3FF';   // Розовый

  // Определяем реальный тип графика и ориентацию осей
  let actualChartType = currentChartType;
  let indexAxis = 'x'; // По умолчанию

  if (currentChartType === 'horizontalBar') {
    actualChartType = 'bar'; // Используем bar, но с горизонтальной ориентацией
    indexAxis = 'y';
  }

  const datasets = [
    {
      label: 'Студент',
      data: studentValues,
      backgroundColor: studentColor,
      borderColor: studentColor,
      borderWidth: 1,
      borderRadius: actualChartType === 'bar' ? 3 : 0
    },
    {
      label: 'Группа',
      data: groupValues,
      backgroundColor: groupColor,
      borderColor: groupColor,
      borderWidth: 1,
      borderRadius: actualChartType === 'bar' ? 3 : 0
    }
  ];

  // Настройка для линейного графика
  if (actualChartType === 'line') {
    datasets.forEach(dataset => {
      dataset.borderWidth = 2;
      dataset.fill = false;
      dataset.pointRadius = 3;
    });
  }

  const config = {
    type: actualChartType, // Используем вычисленный тип
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: indexAxis, // Устанавливаем ориентацию осей
      layout: {
        padding: {
          bottom: 10
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          align: 'center',
          labels: {
            color: fontcolor,
            boxWidth: 16,
            boxHeight: 16,
            padding: 10,
            font: {
              size: 12
            },
            usePointStyle: false
          }
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.parsed.y?.toFixed(2) || '0.00'}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: fontcolor
          },
          grid: {
            color: linecolor
          }
        },
        x: {
          ticks: {
            color: fontcolor
          },
          grid: {
            color: linecolor,
            display: false
          }
        }
      }
    }
  };

  // Устанавливаем высоту canvas перед созданием графика
  const canvas = document.getElementById('p5');

  const ctx = canvas.getContext('2d');
  currentChart = new Chart(ctx, config);
}

  // Event listeners for UI controls
  courseSelect.addEventListener("change", () => {
    const barButton = document.querySelector('.chart-type-btn[data-type="bar"]');

if (barButton) {
  // 1. Убираем active у всех кнопок
  document.querySelectorAll('.chart-type-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  // 2. Добавляем active только к barButton
  barButton.classList.add('active');

  // 3. Обновляем currentChartType (если нужно)
  currentChartType = 'bar';

  // 4. Не вызываем renderChart() — график остаётся как был
}
    loadCourseData(courseSelect.value);
  });

  // Обработчик изменения параметра
parameterSelect.addEventListener('change', () => {
  const selectedOption = parameterSelect.value;
    const barButton = document.querySelector('.chart-type-btn[data-type="bar"]');

if (barButton) {
  // 1. Убираем active у всех кнопок
  document.querySelectorAll('.chart-type-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  // 2. Добавляем active только к barButton
  barButton.classList.add('active');

  // 3. Обновляем currentChartType (если нужно)
  currentChartType = 'bar';

  // 4. Не вызываем renderChart() — график остаётся как был
}
  renderChart(selectedOption);
});


  // Initialize the application
  checkAuth();
});

