const template = document.createElement('template');
template.innerHTML = ` 
'<div id="logo-container">
    <img src="/static/logo.png" alt="Your Logo">
</div>

 <!-- Headings for sub-networks displayed horizontally -->
<div id="sub-networks">
    <a href="/"><h2>Main</h2></a>
    <a href="/help"><h2>Help</h2></a>
    <a href="https://github.com/uzh-dqbm-cmi/BEDICT-V2" target="_blank"><h2>Code</h2></a>
    <a href="/about"><h2>About</h2></a>
</div>`
document.body.appendChild(template.content);
