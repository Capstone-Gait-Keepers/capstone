<template>
  <BasePage v-if="user">
    <h1>Account Settings</h1>
    <h2>Account Information</h2>
    <FormInput label="Name" type="text" v-model="user.name" />
    <FormInput label="Email" type="email" v-model="user.email" />
    <FormInput label="Password" type="password" v-model="user.password" />
    <FormInput label="Sensor ID" type="text" v-model="user.sensor_id" />
    <!-- <FormInput label="Notifications" type="text" v-model="user.notifications" /> -->
    <h2>Visual Display Settings</h2>
    <p>Personalize your display to improve accessibility.</p>
    <FormSwitch label="Large font" id="font" v-model="display.large_font" />
    <FormSwitch label="High contrast" id="contrast" v-model="display.high_contrast" />
    <h2>Health Indicators</h2>
    <p>Choose what information to view on the Home Insights page. If you don't want to see a specific group of insights or recommendations simply turn the toggle off. The system will still collect the data it just won't display it to you so you can turn them back on at any time if you decide you want to see the insights.</p>
    <FormSwitch label="Balance indicators" id="balance" v-model="view_sections.balance" />
    <FormSwitch label="Neurodegenerative disease indicators" id="neurodegenerative" v-model="view_sections.neurodegenerative" />
    <FormSwitch label="Dementia indicators" id="dementia" v-model="view_sections.dementia" />
    <h2>Connected Users</h2>
    <i>Adding family members will allow them to view your individual insights from their own account. Once you add a family member they will receive an email inviting them to create an account and link your data to it. </i>
    <br>
    <div v-for="user_email, i in connected_users" :key="user_email" class="connected-user">
      <p>{{ user_email }}</p>
      <button @click="removeUser(i)">Delete</button>
    </div>
    <FormInput v-if="addingUser" id="new-user" type="email"  v-on:save="openConsent" startEditing />
    <p v-if="successMessage" class="success">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M14 0C6.26801 0 0 6.26801 0 14C0 21.732 6.26801 28 14 28C21.732 28 28 21.732 28 14C28 10.287 26.525 6.72601 23.8995 4.10051C21.274 1.475 17.713 0 14 0ZM12.12 19.3797L7.12 14.4197L8.71 12.8497L12.12 16.2297L19.53 8.87969L21.12 10.4597L12.12 19.3797ZM2 14C2 20.6274 7.37258 26 14 26C20.6274 26 26 20.6274 26 14C26 7.37258 20.6274 2 14 2C7.37258 2 2 7.37258 2 14Z" fill="#68AE21"/>
      </svg>
      {{ successMessage }}
    </p>
    <button @click="startAddingUser">Add Family Member</button>
    <div v-if="showModal" class="consent-modal">
      <div class="consent-modal-content">
        <header><h1>Health Data Sharing Consent Form</h1></header>
        <p>
          I hereby consent to connect a designated family member as a viewer of my health data through
          email. By clicking below, I acknowledge that I understand the purpose of this connection is
          to share my health data with my family member for the purpose of providing them with an
          overview of my health status. I understand that I have the option to revoke this access at any
          time. I acknowledge that while every effort will be made to ensure the security and
          confidentiality of my data, there may be inherent risks associated with sharing personal health
          information electronically. I agree to release WalkWise from any liability arising from the use
          or sharing of my health data. By clicking this consent agreement, I affirm that I have read and
          understand its contents, and I voluntarily agree to connect a family member viewer to my health
          data.
        </p>
        <footer>
          <button @click="addUser">I Agree</button>
          <button @click="resetAddUser">Cancel</button>
        </footer>
      </div>
    </div>
  </BasePage>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import BasePage from '@/components/BasePage.vue';
import FormInput from '@/components/FormInput.vue';
import FormSwitch from '@/components/FormSwitch.vue';
import { store } from '@/store';
const { view_sections, display, user, connected_users } = store;

const addingUser = ref(false);
const successMessage = ref('');
const showModal = ref(false);
const newEmail = ref('');

function startAddingUser() {
  addingUser.value = true;
  successMessage.value = '';
}

function openConsent(email: any) {
  if (typeof email !== 'string') return;
  newEmail.value = email;
  showModal.value = true;
}

function addUser() {
  connected_users.push(newEmail.value);
  successMessage.value = `${newEmail.value} successfully added as an Authorized User.`;
  resetAddUser();
}

function resetAddUser() {
  showModal.value = false;
  addingUser.value = false;
  newEmail.value = '';
}

function removeUser(index: number) {
  const email = connected_users.splice(index, 1);
  successMessage.value = `${email} successfully removed as an Authorized User.`;
}
</script>

<style scoped>
h2 {
  margin-top: .5rem;
}

.connected-user {
  display: flex;
  margin-bottom: 1rem;
}

.connected-user > p {
  border: 1px solid var(--color-main);
  border-radius: 8px;
  padding: 0.3rem 1rem;
  width: 30rem;
  max-width: 100%;
  margin-right: 2rem;
  background-color: white;
}

.connected-user > button {
  background-color: var(--red);
}

.success {
  display: flex;
  color: var(--green);
}

.success svg {
  margin-right: .5rem;
  margin-bottom: 1rem;
  width: 1rem;
}

.consent-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.consent-modal-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  max-width: 600px;
  margin-top: 10vh;
  padding: 1.5rem;
  border: 8px solid var(--color-main);
  border-radius: 16px;
  background-color: white;
}

.consent-modal-content > footer {
  display: flex;
  justify-content: space-around;
  font-size: 1.2em;
  gap: 1rem;
}
</style>