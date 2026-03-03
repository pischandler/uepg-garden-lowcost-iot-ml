(function () {
  const state = {};
  const subscribers = [];

  function updateState(partial) {
    if (!partial || typeof partial !== "object") return;
    Object.keys(partial).forEach(function (k) {
      state[k] = partial[k];
    });
    subscribers.forEach(function (fn) {
      try {
        fn(state);
      } catch (e) {
        console.warn("store subscriber error", e);
      }
    });
  }

  function getState() {
    return state;
  }

  function subscribe(fn) {
    if (typeof fn !== "function") return;
    subscribers.push(fn);
  }

  window.STGStore = { updateState, getState, subscribe };
})();
