import { app } from "../../scripts/app.js";

function createPreviewWidget(node, widget) {
    const container = document.createElement("div");
    container.style.cssText = "position:relative;width:100%;min-height:60px;max-height:200px;overflow:hidden;border-radius:4px;background:#111;border:1px solid #333;display:flex;align-items:center;justify-content:center;";

    const img = document.createElement("img");
    img.style.cssText = "max-width:100%;max-height:200px;object-fit:contain;display:none;";
    img.onerror = () => {
        img.style.display = "none";
        placeholder.style.display = "flex";
        placeholder.textContent = "无法加载预览";
    };

    const placeholder = document.createElement("span");
    placeholder.style.cssText = "color:#555;font-size:11px;pointer-events:none;";
    placeholder.textContent = "输入路径或URL预览图片";

    const loading = document.createElement("span");
    loading.style.cssText = "color:#888;font-size:11px;display:none;";
    loading.textContent = "加载中...";

    const clearBtn = document.createElement("button");
    clearBtn.textContent = "✕";
    clearBtn.style.cssText = "position:absolute;top:2px;right:2px;background:rgba(0,0,0,0.6);color:#aaa;border:none;border-radius:2px;padding:1px 5px;cursor:pointer;font-size:10px;display:none;";
    clearBtn.onclick = () => {
        img.style.display = "none";
        img.src = "";
        placeholder.style.display = "flex";
        placeholder.textContent = "输入路径或URL预览图片";
        clearBtn.style.display = "none";
        node._previewUrl = "";
    };

    container.appendChild(img);
    container.appendChild(placeholder);
    container.appendChild(loading);
    container.appendChild(clearBtn);

    node._previewUrl = "";
    node._previewImg = img;
    node._previewPlaceholder = placeholder;
    node._previewLoading = loading;
    node._previewClearBtn = clearBtn;
    node._previewWidget = widget;

    let debounceTimer = null;
    widget.callback = () => {
        clearTimeout(debounceTimer);
        const val = widget.value?.trim();
        if (!val) {
            img.style.display = "none";
            img.src = "";
            clearBtn.style.display = "none";
            placeholder.style.display = "flex";
            placeholder.textContent = "输入路径或URL预览图片";
            node._previewUrl = "";
            return;
        }
        if (node._previewUrl === val) return;
        debounceTimer = setTimeout(() => loadPreview(node, val), 400);
    };

    return container;
}

async function loadPreview(node, pathOrUrl) {
    const { _previewImg: img, _previewPlaceholder: placeholder, _previewLoading: loading, _previewClearBtn: clearBtn } = node;

    try {
        const resp = await fetch("/image_embeddings/preview_image?path=" + encodeURIComponent(pathOrUrl));
        if (!resp.ok) {
            throw new Error("Failed to load preview");
        }
        const data = await resp.json();
        if (!data.success || !data.data_url) {
            throw new Error(data.error || "Unknown error");
        }
        img.onload = () => {
            img.style.display = "block";
            placeholder.style.display = "none";
            loading.style.display = "none";
            clearBtn.style.display = "block";
        };
        loading.style.display = "block";
        placeholder.style.display = "none";
        img.src = data.data_url;
        node._previewUrl = pathOrUrl;
    } catch {
        img.style.display = "none";
        loading.style.display = "none";
        placeholder.style.display = "flex";
        placeholder.textContent = "无法加载预览";
    }
}

app.registerExtension({
    name: "image_embeddings.CustomImageLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "CustomImageLoader") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            setTimeout(() => {
                const widget = this.widgets?.find(w => w.name === "image_path_or_url");
                if (!widget) return;

                const container = createPreviewWidget(this, widget);
                this.addDOMWidget(nodeData.name, "image_preview", container, {
                    serialize: false,
                });

                if (widget.value?.trim()) {
                    widget.callback?.();
                }
            }, 0);
        };
    },
});
