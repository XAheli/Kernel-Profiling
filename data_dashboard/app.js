const kernelSelect = document.getElementById('kernelSelect');
const metricSelect = document.getElementById('metricSelect');
const tableHead = document.getElementById('tableHead');
const tableBody = document.getElementById('tableBody');
const avgSpeedup = document.getElementById('avgSpeedup');
const maxSpeedup = document.getElementById('maxSpeedup');
const darkModeToggle = document.getElementById('darkModeToggle');
const modeIcon = document.getElementById('modeIcon');
const modeText = document.getElementById('modeText');
const downloadBtn = document.getElementById('downloadBtn');

// Dark mode functionality
function initDarkMode() {
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode === 'true') {
        document.body.classList.add('dark-mode');
        modeIcon.textContent = '☀️';
        modeText.textContent = 'Light';
    }
}

darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
    modeIcon.textContent = isDark ? '☀️' : '🌙';
    modeText.textContent = isDark ? 'Light' : 'Dark';
});

// CSV Download functionality
downloadBtn.addEventListener('click', () => {
    let csvContent = "Kernel,Metric,Configuration,CUDA,Triton,Helion\n";

    for (const [kernelKey, kernelData] of Object.entries(DATA)) {
        const kernelName = kernelKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        for (const [metricKey, metricData] of Object.entries(kernelData.metrics)) {
            const metricName = metricData.name;

            kernelData.configs.forEach((config, idx) => {
                let configName = config.name;
                if (config.detail) {
                    configName += ` (${config.detail})`;
                }

                const cudaVal = metricData.cuda[idx];
                const tritonVal = metricData.triton[idx];
                const helionVal = metricData.helion[idx];

                csvContent += `"${kernelName}","${metricName}","${configName}",${cudaVal},${tritonVal},${helionVal}\n`;
            });
        }
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'gpu_benchmark_complete_data.csv';
    link.click();
});

function populateMetrics() {
    const kernel = kernelSelect.value;
    const metrics = DATA[kernel].metrics;

    metricSelect.innerHTML = '';
    Object.keys(metrics).forEach(metricKey => {
        const option = document.createElement('option');
        option.value = metricKey;
        option.textContent = metrics[metricKey].name;
        metricSelect.appendChild(option);
    });
}

function calculateCellClass(value, values, lowerIsBetter) {
    const sorted = [...values].sort((a, b) => lowerIsBetter ? a - b : b - a);
    const best = sorted[0];
    const second = sorted[1];
    if (value === best) return 'best';
    if (value === second) return 'second';
    return 'worst';
}

function formatValue(value) {
    if (value >= 1e6) return value.toExponential(2);
    if (value >= 1000) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    if (value >= 1) return value.toFixed(2);
    return value.toFixed(4);
}

function calculateSpeedup(values, lowerIsBetter) {
    const sorted = [...values].sort((a, b) => lowerIsBetter ? a - b : b - a);
    return lowerIsBetter ? sorted[2] / sorted[0] : sorted[0] / sorted[2];
}

function getWinner(values, lowerIsBetter) {
    const sorted = [...values].sort((a, b) => lowerIsBetter ? a - b : b - a);
    const best = sorted[0];
    if (values[0] === best) return 'CUDA';
    if (values[1] === best) return 'Triton';
    return 'Helion';
}

function renderTable() {
    const kernel = kernelSelect.value;
    const metric = metricSelect.value;
    const kernelData = DATA[kernel];
    const metricData = kernelData.metrics[metric];

    // Extract unit from metric name
    const metricName = metricData.name;
    let unit = '';
    const unitMatch = metricName.match(/\(([^)]+)\)/);
    if (unitMatch) {
        unit = unitMatch[1];
    } else {
        // For metrics without explicit units in parentheses
        if (metric === 'registers_per_thread') {
            unit = 'registers';
        } else if (metric === 'register_limited_blocks') {
            unit = 'blocks';
        } else if (metric === 'gflops') {
            unit = 'GFLOPs';
        }
    }

    // Only show speedup for duration metric
    const isDuration = metric === 'duration';

    let speedups = [];
    let winners = { CUDA: 0, Triton: 0, Helion: 0 };

    // Build header with conditional speedup column and units
    tableHead.innerHTML = `
        <tr>
            <th>Configuration</th>
            <th>
                <span class="header-impl">CUDA</span>
                ${unit ? `<span class="header-unit">${unit}</span>` : ''}
            </th>
            <th>
                <span class="header-impl">Triton</span>
                ${unit ? `<span class="header-unit">${unit}</span>` : ''}
            </th>
            <th>
                <span class="header-impl">Helion</span>
                ${unit ? `<span class="header-unit">${unit}</span>` : ''}
            </th>
            ${isDuration ? '<th><span class="header-impl">Speedup</span><span class="header-unit">×</span></th>' : ''}
        </tr>
    `;

    tableBody.innerHTML = '';
    kernelData.configs.forEach((config, idx) => {
        const cuda = metricData.cuda[idx];
        const triton = metricData.triton[idx];
        const helion = metricData.helion[idx];
        const values = [cuda, triton, helion];

        const cudaClass = calculateCellClass(cuda, values, metricData.lower_is_better);
        const tritonClass = calculateCellClass(triton, values, metricData.lower_is_better);
        const helionClass = calculateCellClass(helion, values, metricData.lower_is_better);

        let speedupCell = '';
        if (isDuration) {
            const speedup = calculateSpeedup(values, metricData.lower_is_better);
            speedups.push(speedup);

            let speedupClass = 'speedup-medium';
            if (speedup > 50) speedupClass = 'speedup-extreme';
            else if (speedup > 10) speedupClass = 'speedup-high';
            else if (speedup < 1) speedupClass = 'speedup-regression';

            speedupCell = `<td><span class="speedup ${speedupClass}">${speedup.toFixed(1)}x</span></td>`;
        }

        const winner = getWinner(values, metricData.lower_is_better);
        winners[winner]++;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="config-cell">
                ${config.name}
                ${config.detail ? `<span class="config-detail">${config.detail}</span>` : ''}
            </td>
            <td class="${cudaClass}">${formatValue(cuda)}</td>
            <td class="${tritonClass}">${formatValue(triton)}</td>
            <td class="${helionClass}">${formatValue(helion)}</td>
            ${speedupCell}
        `;
        tableBody.appendChild(row);
    });

    // Update stats - show speedup stats only for duration
    if (isDuration && speedups.length > 0) {
        const avgSpeed = speedups.reduce((a, b) => a + b, 0) / speedups.length;
        const maxSpeed = Math.max(...speedups);
        avgSpeedup.textContent = avgSpeed.toFixed(1) + 'x';
        maxSpeedup.textContent = maxSpeed.toFixed(1) + 'x';
        avgSpeedup.parentElement.style.display = 'block';
        maxSpeedup.parentElement.style.display = 'block';
    } else {
        avgSpeedup.parentElement.style.display = 'none';
        maxSpeedup.parentElement.style.display = 'none';
    }
}

kernelSelect.addEventListener('change', () => {
    populateMetrics();
    renderTable();
});

metricSelect.addEventListener('change', renderTable);

initDarkMode();
populateMetrics();
renderTable();
