export default class LineDrawer {
    constructor(canvasId, videoId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.video = document.getElementById(videoId);
        this.wrapper = this.canvas.parentElement;

        this.mode = 'none';
        this.isDrawing = false;
        this.isDragging = false;
        this.dragPoint = null;

        this.point1 = null;
        this.point2 = null;
        this.tempPoint = null;

        this.lineColor = '#00ff00';
        this.lineWidth = 3;
        this.pointRadius = 8;
        this.nearThreshold = 15;

        this.videoWidth = 0;
        this.videoHeight = 0;

        this._bindEvents();
    }

    _bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this._onMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this._onMouseLeave(e));

        this.video.addEventListener('load', () => this._updateVideoSize());
        window.addEventListener('resize', () => this._resizeCanvas());
        document.addEventListener('mousemove', (e) => this._onMouseMove(e));
        document.addEventListener('mouseup', (e) => this._onMouseUp(e));
        setInterval(() => this._updateVideoSize(), 2000);
    }

    _updateVideoSize() {
        if (this.video.naturalWidth > 0) {
            this.videoWidth = this.video.naturalWidth;
            this.videoHeight = this.video.naturalHeight;
            this._resizeCanvas();
        }
    }

    _resizeCanvas() {
        const rect = this.wrapper.getBoundingClientRect();
        const targetW = Math.max(1, Math.round(rect.width));
        const targetH = Math.max(1, Math.round(rect.height));
        if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
            this.canvas.width = targetW;
            this.canvas.height = targetH;
            this._draw();
        }
    }

    _getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    _getDisplayRect() {
        const canvasW = this.canvas.width;
        const canvasH = this.canvas.height;
        if (!canvasW || !canvasH || !this.videoWidth || !this.videoHeight) {
            return { offsetX: 0, offsetY: 0, displayW: canvasW, displayH: canvasH };
        }

        const videoAspect = this.videoWidth / this.videoHeight;
        const canvasAspect = canvasW / canvasH;

        let displayW, displayH, offsetX, offsetY;
        if (videoAspect > canvasAspect) {
            displayW = canvasW;
            displayH = canvasW / videoAspect;
            offsetX = 0;
            offsetY = (canvasH - displayH) / 2;
        } else {
            displayH = canvasH;
            displayW = canvasH * videoAspect;
            offsetX = (canvasW - displayW) / 2;
            offsetY = 0;
        }

        return { offsetX, offsetY, displayW, displayH };
    }

    _isInDisplayArea(pt) {
        const { offsetX, offsetY, displayW, displayH } = this._getDisplayRect();
        return pt.x >= offsetX && pt.x <= (offsetX + displayW) && pt.y >= offsetY && pt.y <= (offsetY + displayH);
    }

    _clampToDisplayArea(pt) {
        const { offsetX, offsetY, displayW, displayH } = this._getDisplayRect();
        return {
            x: Math.max(offsetX, Math.min(offsetX + displayW, pt.x)),
            y: Math.max(offsetY, Math.min(offsetY + displayH, pt.y))
        };
    }

    _canvasToImage(cx, cy) {
        if (this.videoWidth === 0 || this.videoHeight === 0) return [Math.round(cx), Math.round(cy)];

        const { offsetX, offsetY, displayW, displayH } = this._getDisplayRect();
        const imgX = Math.round((cx - offsetX) / displayW * this.videoWidth);
        const imgY = Math.round((cy - offsetY) / displayH * this.videoHeight);

        return [
            Math.max(0, Math.min(this.videoWidth, imgX)),
            Math.max(0, Math.min(this.videoHeight, imgY))
        ];
    }

    _imageToCanvas(imgX, imgY) {
        if (this.videoWidth === 0 || this.videoHeight === 0) return { x: imgX, y: imgY };

        const { offsetX, offsetY, displayW, displayH } = this._getDisplayRect();
        return {
            x: offsetX + (imgX / this.videoWidth) * displayW,
            y: offsetY + (imgY / this.videoHeight) * displayH
        };
    }

    _isNearPoint(canvasPoint, endPoint) {
        if (!endPoint) return false;
        const dx = canvasPoint.x - endPoint.x;
        const dy = canvasPoint.y - endPoint.y;
        return Math.sqrt(dx * dx + dy * dy) < this.nearThreshold;
    }

    _onMouseDown(e) {
        if (e.target !== this.canvas) return;
        e.preventDefault();
        const pt = this._getCanvasPoint(e);

        if (this.mode === 'draw') {
            if (!this._isInDisplayArea(pt)) return;
            this.isDrawing = true;
            this.point1 = pt;
            this.point2 = null;
            this.tempPoint = pt;
            this._draw();
        } else if (this.mode === 'edit' && this.point1 && this.point2) {
            if (this._isNearPoint(pt, this.point1)) {
                this.isDragging = true;
                this.dragPoint = 1;
                this.canvas.style.cursor = 'grabbing';
            } else if (this._isNearPoint(pt, this.point2)) {
                this.isDragging = true;
                this.dragPoint = 2;
                this.canvas.style.cursor = 'grabbing';
            }
        }
    }

    _onMouseMove(e) {
        if (!this.isDrawing && !this.isDragging && e.target !== this.canvas) return;
        if ((this.isDrawing || this.isDragging) && e.cancelable) e.preventDefault();
        const rawPt = this._getCanvasPoint(e);

        if (this.mode === 'draw' && this.isDrawing) {
            this.tempPoint = this._clampToDisplayArea(rawPt);
            this._draw();
        } else if (this.mode === 'edit' && this.isDragging) {
            const pt = this._clampToDisplayArea(rawPt);
            if (this.dragPoint === 1) this.point1 = pt;
            else if (this.dragPoint === 2) this.point2 = pt;
            this._draw();
        } else if (this.mode === 'edit' && this.point1 && this.point2) {
            if (this._isNearPoint(rawPt, this.point1) || this._isNearPoint(rawPt, this.point2)) {
                this.canvas.style.cursor = 'grab';
            } else {
                this.canvas.style.cursor = 'default';
            }
        }
    }

    _onMouseUp(e) {
        if ((this.isDrawing || this.isDragging) && e.cancelable) e.preventDefault();
        const pt = this._clampToDisplayArea(this._getCanvasPoint(e));

        if (this.mode === 'draw' && this.isDrawing) {
            this.isDrawing = false;
            this.point2 = pt;
            this.tempPoint = null;
            this.mode = 'none';
            this.canvas.style.cursor = 'default';
            this.canvas.style.pointerEvents = 'none';
            this.canvas.style.display = 'none';
            this._notifyLineSet();
        } else if (this.mode === 'edit' && this.isDragging) {
            this.isDragging = false;
            this.dragPoint = null;
            this.canvas.style.cursor = 'default';
            this._draw();
            this._notifyLineSet();
        }
    }

    _onMouseLeave() {
        if (this.mode === 'draw' && this.isDrawing) {
            this._draw();
        }
    }

    _draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const p1 = this.point1;
        const p2 = this.isDrawing ? this.tempPoint : this.point2;

        if (!p1 || !p2) return;

        this.ctx.beginPath();
        this.ctx.moveTo(p1.x, p1.y);
        this.ctx.lineTo(p2.x, p2.y);
        this.ctx.strokeStyle = this.lineColor;
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.stroke();

        this._drawEndpoint(p1, 'A', '#ff4444');
        this._drawEndpoint(p2, 'B', '#4444ff');
    }

    _drawEndpoint(pt, label, color) {
        this.ctx.beginPath();
        this.ctx.arc(pt.x, pt.y, this.pointRadius, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 12px Inter, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(label, pt.x, pt.y);
    }

    startDraw() {
        // 即时获取视频实际尺寸，避免依赖异步 load 事件或轮询
        this._updateVideoSize();
        if (this.videoWidth === 0 || this.videoHeight === 0) {
            console.warn('无法绘制：视频画面尚未加载，请稍后再试');
            return false;
        }
        this.mode = 'draw';
        this.point1 = null;
        this.point2 = null;
        this.tempPoint = null;
        this.canvas.style.display = 'block';
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'crosshair';
        this._resizeCanvas();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        return true;
    }

    startEdit() {
        if (!this.point1 || !this.point2) return false;
        this.mode = 'edit';
        this.canvas.style.display = 'block';
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'default';
        this._resizeCanvas();
        this._draw();
        return true;
    }

    stopEdit() {
        this.mode = 'none';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.canvas.style.display = 'none';
    }

    clear() {
        this.point1 = null;
        this.point2 = null;
        this.tempPoint = null;
        this.mode = 'none';
        this.canvas.style.display = 'none';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.cursor = 'default';
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    setLineFromImage(imgP1, imgP2) {
        this._updateVideoSize();
        this.point1 = this._imageToCanvas(imgP1[0], imgP1[1]);
        this.point2 = this._imageToCanvas(imgP2[0], imgP2[1]);
        this.canvas.style.display = 'none';
        this.canvas.style.pointerEvents = 'none';
    }

    getImagePoints() {
        if (!this.point1 || !this.point2) return null;
        return {
            point1: this._canvasToImage(this.point1.x, this.point1.y),
            point2: this._canvasToImage(this.point2.x, this.point2.y)
        };
    }

    _notifyLineSet() {
        const pts = this.getImagePoints();
        if (pts && this.onLineSet) {
            this.onLineSet(pts.point1, pts.point2);
        }
    }
}

