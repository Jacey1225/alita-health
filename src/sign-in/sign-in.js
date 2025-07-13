async function loginUser() {
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  const response = await fetch('/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });

  const data = await response.json();

  if (response.ok) {
    localStorage.setItem('username', username); // ✅ store username
    window.location.href = '../dashboard/dashboard.html'; // ✅ redirect to dashboard
  } else {
    alert(data.error || 'Login failed');
  }
}
