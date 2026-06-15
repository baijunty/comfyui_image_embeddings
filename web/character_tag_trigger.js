import { app } from "../../scripts/app.js";

function createPopup() {
    const popup = document.createElement("div");
    popup.style.cssText = "position:fixed;z-index:9999;background:#1e1e1e;border:1px solid #555;border-radius:6px;min-width:280px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.5);font-family:sans-serif;";

    const inner = document.createElement("div");
    inner.style.padding = "8px";

    const typeContainer = document.createElement("div");
    typeContainer.style.cssText = "display:flex;gap:4px;margin-bottom:6px;";

    const charBtn = document.createElement("button");
    charBtn.textContent = "角色";
    charBtn.style.cssText = "flex:1;padding:4px;background:#333;border:1px solid #777;color:#ccc;border-radius:3px;cursor:pointer;";
    
    const artistBtn = document.createElement("button");
    artistBtn.textContent = "画师";
    artistBtn.style.cssText = "flex:1;padding:4px;background:#2a2a2a;border:1px solid #555;color:#ccc;border-radius:3px;cursor:pointer;";

    charBtn.onclick = () => {
        popup._searchType = "characters";
        charBtn.style.background = "#333";
        charBtn.style.borderColor = "#777";
        artistBtn.style.background = "#2a2a2a";
        artistBtn.style.borderColor = "#555";
        popup._input.placeholder = "输入角色名搜索...";
        popup._input.value = "";
        popup._results.innerHTML = "";
    };

    artistBtn.onclick = () => {
        popup._searchType = "artists";
        artistBtn.style.background = "#333";
        artistBtn.style.borderColor = "#777";
        charBtn.style.background = "#2a2a2a";
        charBtn.style.borderColor = "#555";
        popup._input.placeholder = "输入画师名搜索...";
        popup._input.value = "";
        popup._results.innerHTML = "";
    };

    typeContainer.appendChild(charBtn);
    typeContainer.appendChild(artistBtn);

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "输入角色名搜索...";
    input.style.cssText = "width:100%;padding:6px;background:#2a2a2a;border:1px solid #555;color:#fff;border-radius:3px;box-sizing:border-box;outline:none;";

    const results = document.createElement("div");
    results.style.cssText = "max-height:200px;overflow-y:auto;margin-top:4px;";

    inner.appendChild(typeContainer);
    inner.appendChild(input);
    inner.appendChild(results);
    popup.appendChild(inner);

    popup._input = input;
    popup._results = results;
    popup._searchType = "characters";

    let debounceTimer = null;

    input.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        const q = input.value.trim();
        if (q.length > 0) {
            debounceTimer = setTimeout(() => doSearch(popup, q), 300);
        } else {
            results.innerHTML = "";
        }
    });

    input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") popup.style.display = "none";
    });

    document.addEventListener("mousedown", (e) => {
        if (popup.style.display !== "none" && !popup.contains(e.target)) {
            popup.style.display = "none";
        }
    });

    document.body.appendChild(popup);
    return popup;
}

async function doSearch(popup, query) {
    const results = popup._results;
    const searchType = popup._searchType;
    try {
        let items = [];
        if (searchType === "artists") {
            const r = await fetch("/image_embeddings/search_artists?query=" + encodeURIComponent(query));
            if (r.ok) {
                items = (await r.json()).map(item => ({...item, type: 'artist'}));
            }
        } else {
            const r = await fetch("/image_embeddings/search_characters?query=" + encodeURIComponent(query));
            if (r.ok) {
                items = (await r.json()).map(item => ({...item, type: 'character'}));
            }
        }
        
        results.innerHTML = "";
        if (!items.length) {
            results.innerHTML = '<div style="padding:8px;color:#888;font-size:12px;">未找到匹配</div>';
            return;
        }
        
        items.forEach((item) => {
            const d = document.createElement("div");
            if (item.type === 'artist') {
                d.textContent = item.name;
                d.style.color = "#8ab";
            } else {
                d.textContent = item.cn_name ? `${item.name} (${item.cn_name})` : item.name;
                d.style.color = "#ccc";
            }
            d.style.cssText = "padding:8px 10px;cursor:pointer;font-size:12px;border-bottom:1px solid #2a2a2a;";
            d.onmouseenter = () => d.style.background = "#333";
            d.onmouseleave = () => d.style.background = "transparent";
            d.onmousedown = (e) => {
                e.preventDefault();
                if (popup._searchType === "artists" && popup._artistWidget) {
                    popup._artistWidget.value = item.name;
                } else if (popup._searchType === "characters" && popup._characterWidget) {
                    popup._characterWidget.value = item.name;
                }
                if (app.canvas) app.canvas.setDirty(true);
                popup.style.display = "none";
            };
            results.appendChild(d);
        });
    } catch {
        results.innerHTML = '<div style="padding:8px;color:#888;font-size:12px;">查询失败</div>';
    }
}

let popup = null;

app.registerExtension({
    name: "image_embeddings.CharacterTagTrigger",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "CharacterTagTrigger") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            setTimeout(() => {
                const characterWidget = this.widgets?.find(w => w.name === "character_name");
                const artistWidget = this.widgets?.find(w => w.name === "artist_name");

                if (!popup) popup = createPopup();

                if (!this.widgets?.find(w => w.name === "🔍 搜索")) {
                    this.addWidget("button", "🔍 搜索", "", () => {
                        popup._characterWidget = characterWidget;
                        popup._artistWidget = artistWidget;
                        
                        const rect = app.canvas.canvas.getBoundingClientRect();
                        const scale = app.canvas.ds.scale;
                        const ox = app.canvas.ds.offset?.[0] || 0;
                        const oy = app.canvas.ds.offset?.[1] || 0;
                        const x = (this.pos[0] + this.size[0] / 2) * scale + ox + rect.left;
                        const y = (this.pos[1] + this.size[1]) * scale + oy + rect.top;
                        popup.style.left = Math.max(10, Math.min(x - 140, window.innerWidth - 300)) + "px";
                        popup.style.top = Math.max(10, y + 5) + "px";
                        popup.style.display = "block";
                        popup._input.value = "";
                        popup._input.placeholder = "输入角色名搜索...";
                        popup._input.focus();
                        popup._results.innerHTML = "";
                        
                        const charBtn = popup.querySelector("button:first-child");
                        const artistBtn = popup.querySelector("button:last-child");
                        if (charBtn && artistBtn) {
                            charBtn.style.background = "#333";
                            charBtn.style.borderColor = "#777";
                            artistBtn.style.background = "#2a2a2a";
                            artistBtn.style.borderColor = "#555";
                        }
                        popup._searchType = "characters";
                    }, { serialize: false });
                }

                if (app.canvas) app.canvas.setDirty(true);
            }, 0);
        };
    },
});
