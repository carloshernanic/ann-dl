// Configuração do MathJax 3. Este arquivo é carregado ANTES da biblioteca
// (ver extra_javascript em mkdocs.yml), por isso define window.MathJax.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};

// Material usa navegação instantânea (SPA): a cada troca de página é preciso
// re-typesetar. O primeiro disparo de document$ acontece antes de o MathJax
// existir — nesse caso o próprio MathJax faz o typeset inicial. Nos seguintes,
// encadeamos na promessa de inicialização para nunca rodar dois typesets ao
// mesmo tempo (o que causava "Cannot read properties of null (replaceChild)").
document$.subscribe(() => {
  if (!window.MathJax || !window.MathJax.startup || !window.MathJax.startup.promise) {
    return;
  }
  window.MathJax.startup.promise = window.MathJax.startup.promise
    .then(() => {
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      return window.MathJax.typesetPromise();
    })
    .catch((err) => console.warn("MathJax: falha ao re-typesetar", err));
});
