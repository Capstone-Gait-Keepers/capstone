<template>
  <main>
    <Logo style="margin: 1rem" />
    <div class="content">
      <h1>Welcome to WalkWise.</h1>
      <h3>What do you want to do today?</h3>
      <div class="options">
        <button style="background-color: var(--color-main)">
          <RouterLink to="/">Log In</RouterLink>
        </button>
        <p>or</p>
        <button style="background-color: var(--green)">Sign Up</button>
      </div>
    </div>
  </main>
</template>

<script setup>
import Logo from '@/components/Logo.vue'; 
</script>

<style scoped>
.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20vh;
  font-size: 1.5rem;
  gap: 1rem;
}

h1 {
  text-align: center;
  font-size: 4rem;
}

.options {
  display: flex;
  align-items: center;
  margin-top: 2rem;
  gap: 1rem;
}

button a {
  color: white;
  text-decoration: none;
  background-color: transparent;
  padding: 0;
}
</style>