<template>
  <main>
    <Logo />
    <div class="center">
      <div class="modal">
        <header>
          <h1>Information Consent</h1>
        </header>
        <div class="body">
          <p>
            I hereby consent to participate in the health monitoring technology program offered by WalkWise. By signing below,
            I acknowledge that I understand the purpose of this program is to monitor and track my health data for the purpose
            of improving my overall wellness and providing peace of mind to my family members. I understand that I have the
            option to share my health data with designated family members or caregivers. I acknowledge that while every effort
            will be made to ensure the security and confidentiality of my data, there may be inherent risks associated with
            sharing personal health information electronically. I agree to release WalkWise from any liability arising from the
            use or sharing of my health data. I understand that I may withdraw my consent at any time by contacting
            support@walkwise.ca. By signing this consent agreement, I affirm that I have read and understand its contents, and
            I voluntarily agree to participate in the health monitoring technology program.
          </p>
        </div>
        <footer>
          <button><RouterLink to="/">Agree</RouterLink></button>
          <button><RouterLink to="/login">Cancel</RouterLink></button>
        </footer>
      </div>
    </div>
  </main>
</template>

<script setup>
import { RouterLink } from 'vue-router';

import Logo from '@/components/Logo.vue';
</script>

<style scoped>
main {
  margin: 1rem;
}

.center {
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal {
  max-width: 600px;
  /* Center of screen, both dims using flex*/
  margin-top: 10vh;
  padding: 1.5rem;
  border: 8px solid var(--color-main);
  border-radius: 16px;
  background-color: white;  
}

header {
  margin-bottom: 20px;
}

header h1 {
  text-align: center;
  font-size: 24px;
}

.body {
  margin-bottom: 20px;
  text-align: justify;
}

footer {
  display: flex;
  justify-content: space-around;
  font-size: 1.2rem;
}
</style>
