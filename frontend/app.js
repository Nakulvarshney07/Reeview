document.addEventListener("DOMContentLoaded", () => {
    loadSampleProducts();
});

function switchTab(tabName) {
    document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
    document.querySelectorAll(".content-area").forEach(area => area.classList.add("hidden"));

    if (tabName === 'predict') {
        document.getElementById("tab-predict").classList.add("active");
        document.getElementById("view-predict").classList.remove("hidden");
    } else if (tabName === 'benchmark') {
        document.getElementById("tab-benchmark").classList.add("active");
        document.getElementById("view-benchmark").classList.remove("hidden");
        loadBenchmark();
    }
}

async function loadSampleProducts() {
    const chipGroup = document.getElementById("sample-chips");
    chipGroup.innerHTML = "<span style='font-size:12px; color:#9ca3af;'>Loading products...</span>";

    try {
        const res = await fetch("/api/products");
        const data = await res.json();

        if (data.status === "success" && data.products) {
            chipGroup.innerHTML = "";
            data.products.forEach(p => {
                const chip = document.createElement("button");
                chip.className = "sample-chip";
                chip.innerText = p.name.split(" ")[0] + " " + (p.name.split(" ")[1] || "");
                chip.title = p.name;
                chip.onclick = () => loadProductDetail(p.id);
                chipGroup.appendChild(chip);
            });
        }
    } catch (err) {
        console.error("Failed to load sample products:", err);
        chipGroup.innerHTML = "<span style='font-size:12px; color:#ef4444;'>Failed to load samples</span>";
    }
}

async function loadProductDetail(productId) {
    try {
        const res = await fetch(`/api/product/${productId}`);
        const data = await res.json();
        if (data.status === "success" && data.data) {
            const p = data.data;
            document.getElementById("product-input").value = p.combined_text || "";
        }
    } catch (err) {
        console.error("Error fetching product detail:", err);
    }
}

async function runPrediction() {
    const rawText = document.getElementById("product-input").value.strip ? document.getElementById("product-input").value.strip() : document.getElementById("product-input").value.trim();
    if (!rawText) {
        alert("Please paste product title, specifications, and reviews into the input box.");
        return;
    }

    const btn = document.getElementById("btn-predict");
    const spinner = document.getElementById("predict-spinner");
    const btnText = btn.querySelector(".btn-text");

    btn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.innerText = "Analyzing Text & Predicting...";

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ raw_text: rawText })
        });

        const data = await res.json();

        if (data.status === "success" && data.predictions) {
            renderPredictions(data.predictions);
        } else {
            alert(data.message || "Failed to predict aspects.");
        }
    } catch (err) {
        console.error("Error in prediction request:", err);
        alert("Error connecting to server.");
    } finally {
        btn.disabled = false;
        spinner.classList.add("hidden");
        btnText.innerText = "Predict Aspects & Sub-aspects";
    }
}

function renderPredictions(predictions) {
    const placeholder = document.getElementById("output-placeholder");
    const container = document.getElementById("output-results");
    const list = document.getElementById("aspects-list");
    const countBadge = document.getElementById("predicted-count");

    placeholder.classList.add("hidden");
    container.classList.remove("hidden");
    list.innerHTML = "";

    countBadge.innerText = `${predictions.length} Experience Aspects Predicted`;

    predictions.forEach(item => {
        const card = document.createElement("div");
        card.className = "aspect-card";

        const confidencePct = Math.round((item.confidence || 0.85) * 100);
        const subaspectsHtml = (item.sub_aspects || [])
            .map(sub => `<span class="subaspect-tag">${sub}</span>`)
            .join("");

        card.innerHTML = `
            <div class="aspect-header">
                <span class="aspect-title">${item.aspect}</span>
                <span class="aspect-confidence">${confidencePct}% Confidence</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: ${confidencePct}%"></div>
            </div>
            <div class="subaspects-container">
                ${subaspectsHtml}
            </div>
        `;

        list.appendChild(card);
    });
}

async function loadBenchmark() {
    const tbody = document.getElementById("benchmark-tbody");
    tbody.innerHTML = `<tr><td colspan="4" class="loading-td" style="text-align:center; padding:30px; color:#9ca3af;">Computing benchmark evaluation on unseen test products...</td></tr>`;

    try {
        const res = await fetch("/api/benchmark");
        const data = await res.json();

        if (data.status === "success" && data.benchmark) {
            const b = data.benchmark;
            const our = b.our_model_summary || {};
            const llama = b.llama_baseline_summary || {};

            const rows = [
                { name: "Aspect Precision", key: "aspect_precision" },
                { name: "Aspect Recall", key: "aspect_recall" },
                { name: "Aspect F1 Score", key: "aspect_f1" },
                { name: "Sub-aspect Precision", key: "subaspect_precision" },
                { name: "Sub-aspect Recall", key: "subaspect_recall" },
                { name: "Sub-aspect F1 Score", key: "subaspect_f1" },
                { name: "Invalid Aspect Rate", key: "invalid_aspect_rate" },
                { name: "Redundancy Rate", key: "redundancy_rate" },
                { name: "Human Alignment Score", key: "human_alignment" }
            ];

            tbody.innerHTML = "";
            rows.forEach(r => {
                const tr = document.createElement("tr");
                const ourVal = our[r.key] !== undefined ? (our[r.key] * 100).toFixed(1) + "%" : "N/A";
                const llamaVal = llama[r.key] !== undefined ? (llama[r.key] * 100).toFixed(1) + "%" : "N/A";

                tr.innerHTML = `
                    <td class="metric-name">${r.name}</td>
                    <td>100.0% (Ground Truth)</td>
                    <td>${llamaVal}</td>
                    <td class="highlight-val">${ourVal}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error("Failed to load benchmark:", err);
        tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:#ef4444; padding:20px;">Failed to load benchmark data.</td></tr>`;
    }
}
