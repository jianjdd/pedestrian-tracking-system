export const appState = {
  currentVideoSource: null,
  poller: null,
  chartCounter: 0,
};

export function setCurrentVideoSource(source) {
  appState.currentVideoSource = source;
}

